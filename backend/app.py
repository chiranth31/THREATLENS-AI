from pathlib import Path
import json, math, os, sqlite3, hashlib, hmac, secrets, time, re
from functools import wraps
from datetime import timedelta
from flask import Flask, jsonify, request, send_from_directory, session
try:
    from .feature_engineering import extract_url_features
except ImportError:
    from feature_engineering import extract_url_features
try:
    from .database import db, init_db, is_integrity_error, USE_POSTGRES
except ImportError:
    from database import db, init_db, is_integrity_error, USE_POSTGRES

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
MODELS = ROOT / "models"
DATA = ROOT / "data"
BUNDLE_PATH = MODELS / "model_bundle.json"
METRICS_PATH = MODELS / "metrics.json"
if not USE_POSTGRES:
    DATA.mkdir(exist_ok=True)
SECRET_PATH = DATA / ".session_secret"


app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")

def load_or_create_secret():
    env = os.environ.get("THREATLENS_SECRET")
    if env:
        return env
    if SECRET_PATH.exists():
        return SECRET_PATH.read_text(encoding="utf-8").strip()
    # Local development convenience only. Vercel must use THREATLENS_SECRET.
    if os.environ.get("VERCEL"):
        raise RuntimeError("THREATLENS_SECRET environment variable is required on Vercel.")
    secret = secrets.token_urlsafe(48)
    SECRET_PATH.write_text(secret, encoding="utf-8")
    try: os.chmod(SECRET_PATH, 0o600)
    except Exception: pass
    return secret

app.secret_key = load_or_create_secret()
app.permanent_session_lifetime = timedelta(hours=8)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("THREATLENS_HTTPS", "0") == "1",
    MAX_CONTENT_LENGTH=64 * 1024,
)

LOGIN_MAX_FAILURES = 5
LOGIN_LOCK_MINUTES = 15
PASSWORD_MIN = 12
PBKDF2_ITERATIONS = 600_000


def get_db():
    return db(ROOT)


def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    try:
        parts = stored.split("$")
        if len(parts) == 4 and parts[0] == "pbkdf2":
            iterations = int(parts[1]); salt = bytes.fromhex(parts[2]); expected = parts[3]
        else:
            # Backward compatibility with the earlier local prototype format.
            salt_hex, expected = stored.split(":", 1); salt = bytes.fromhex(salt_hex); iterations = 120000
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations).hex()
        return hmac.compare_digest(digest, expected)
    except Exception:
        return False


def password_policy(password):
    if len(password) < PASSWORD_MIN: return False, f"Password must be at least {PASSWORD_MIN} characters."
    if len(password) > 128: return False, "Password must be 128 characters or fewer."
    return True, ""


def create_user(con, name, email, password, role="Analyst"):
    ok, msg = password_policy(password)
    if not ok: raise ValueError(msg)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    con.execute("INSERT INTO users(name,email,password_hash,role,created_at,password_changed_at) VALUES(?,?,?,?,?,?)",
                (name.strip(), email.strip().lower(), hash_password(password), role, now, now))


def load_json(path, default):
    if path.exists():
        try: return json.loads(path.read_text(encoding="utf-8"))
        except Exception: pass
    return default


def load_bundle(): return load_json(BUNDLE_PATH, None)

def load_metrics():
    return load_json(METRICS_PATH, {"trained": False, "message": "Models are not exported yet."})


def audit(event, actor_id=None, target_id=None, detail=""):
    try:
        con = get_db(); con.execute("INSERT INTO audit_log(actor_user_id,event,target_user_id,detail,ip,created_at) VALUES(?,?,?,?,?,datetime('now','localtime'))",
            (actor_id, event, target_id, detail, request.remote_addr or "local")); con.commit(); con.close()
    except Exception: pass


def public_user(u):
    return {"id":u["id"],"name":u["name"],"email":u["email"],"role":u["role"],"status":u.get("status", "Active") if isinstance(u, dict) else u["status"],
            "created_at":u["created_at"],"last_login_at":u["last_login_at"],"mfa_enabled":bool(u["mfa_enabled"])}


def current_user():
    uid = session.get("user_id"); version = session.get("session_version")
    if not uid: return None
    con=get_db(); row=con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone(); con.close()
    if not row or row["status"] != "Active": return None
    if version is not None and int(row["session_version"]) != int(version): return None
    return dict(row)


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user(): return jsonify({"error":"Authentication required. Please sign in again."}),401
        return fn(*args, **kwargs)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        u=current_user()
        if not u: return jsonify({"error":"Authentication required. Please sign in again."}),401
        if u["role"] != "Administrator": return jsonify({"error":"Administrator access required."}),403
        return fn(*args, **kwargs)
    return wrapper


def csrf_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = request.headers.get("X-CSRF-Token", "")
        expected = session.get("csrf_token")
        if not expected or not token or not hmac.compare_digest(token, expected):
            return jsonify({"error":"Security token expired. Refresh the page and try again."}),403
        return fn(*args, **kwargs)
    return wrapper


def sigmoid(z):
    if z >= 0:
        e=math.exp(-z); return 1.0/(1.0+e)
    e=math.exp(z); return e/(1.0+e)


def tree_predict(nodes,x):
    i=0
    while True:
        n=nodes[i]
        if n["left"] == -1 and n["right"] == -1:
            vals=n["value"]; total=sum(vals) or 1.0
            return float(vals[1]/total) if len(vals)>1 else 0.0
        f=n["feature"]; i=n["left"] if x[f] <= n["threshold"] else n["right"]


def rf_predict(model,x):
    probs=[tree_predict(t,x) for t in model["trees"]]
    return sum(probs)/len(probs) if probs else 0.0


def lr_predict(model,x): return sigmoid(model["intercept"]+sum(w*v for w,v in zip(model["weights"],x)))

def predict_probability(bundle,name,x):
    m=bundle["models"][name]
    if m["kind"]=="logistic_regression": return lr_predict(m,x)
    if m["kind"]=="decision_tree": return tree_predict(m["nodes"],x)
    return rf_predict(m,x)


@app.get("/")
def index():
    response = send_from_directory(FRONTEND,"index.html")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response

@app.get("/api/csrf")
def csrf():
    session.setdefault("csrf_token", secrets.token_urlsafe(32))
    return jsonify({"token":session["csrf_token"]})

@app.get("/api/health")
def health():
    m=load_metrics(); return jsonify({"ok":True,"trained":BUNDLE_PATH.exists(),"authenticated":bool(current_user()),"metrics":m})

@app.get("/api/metrics")
def metrics(): return jsonify(load_metrics())

@app.get("/api/auth/status")
def auth_status():
    con=get_db(); count=con.execute("SELECT COUNT(*) FROM users").fetchone()[0]; con.close()
    return jsonify({"initialized":count>0,"user_count":count,"setup_available":count==0})

@app.post("/api/auth/initialize")
@csrf_required
def initialize_workspace():
    data=request.get_json(silent=True) or {}
    name=str(data.get("name","ThreatLens Administrator")).strip()
    email=str(data.get("email","")).strip().lower()
    password=str(data.get("password",""))
    if len(name)<2 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+",email):
        return jsonify({"error":"Enter a valid administrator name and email."}),400
    ok,msg=password_policy(password)
    if not ok: return jsonify({"error":msg}),400
    con=get_db(); count=con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count>0:
        con.close(); return jsonify({"error":"This workspace is already initialized. Only an administrator can create additional users."}),409
    try:
        create_user(con,name,email,password,"Administrator"); con.commit(); row=con.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone()
    except Exception as e:
        if is_integrity_error(e):
            con.close(); return jsonify({"error":"An account with this email already exists."}),409
        raise
    except ValueError as e:
        con.close(); return jsonify({"error":str(e)}),400
    con.close()
    session.clear(); session.permanent=True; session["user_id"]=row["id"]; session["session_version"]=row["session_version"]; session["csrf_token"]=secrets.token_urlsafe(32)
    audit("WORKSPACE_INITIALIZED", target_id=row["id"], detail="First administrator created")
    return jsonify({"ok":True,"user":public_user(dict(row)),"csrf":session["csrf_token"]})

@app.post("/api/auth/load-demo")
@csrf_required
def load_demo_accounts():
    con=get_db(); count=con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count>0:
        con.close(); return jsonify({"error":"This workspace is already initialized. Demo bootstrap is only available on a fresh installation."}),409
    create_user(con,"ThreatLens Administrator","admin@threatlens.ai","ThreatLens!2026_Admin","Administrator")
    create_user(con,"Security Analyst","analyst@threatlens.ai","ThreatLens!2026_Analyst","Analyst")
    con.commit(); con.close()
    audit("DEMO_ACCOUNTS_LOADED", detail="Hackathon administrator and analyst accounts created")
    return jsonify({"ok":True,"message":"Hackathon demo accounts loaded. You can now sign in as Administrator or Analyst."})

@app.get("/api/auth/me")
def me():
    u=current_user(); return jsonify({"authenticated":bool(u),"user":public_user(u) if u else None})

@app.post("/api/auth/login")
@csrf_required
def login():
    data=request.get_json(silent=True) or {}; email=str(data.get("email","")).strip().lower(); password=str(data.get("password",""))
    con=get_db(); row=con.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone()
    if not row:
        con.close(); time.sleep(0.15); audit("LOGIN_FAILURE", detail="Unknown account")
        return jsonify({"error":"Invalid email or password."}),401
    now=time.time()
    locked_until=row["locked_until"]
    if locked_until:
        try:
            if float(locked_until) > now:
                remaining=max(1,int(float(locked_until)-now)//60+1); con.close()
                audit("LOGIN_BLOCKED", target_id=row["id"], detail="Account temporarily locked")
                return jsonify({"error":f"Account temporarily locked. Try again in about {remaining} minute(s) or contact an administrator."}),423
        except ValueError: pass
    if row["status"] != "Active":
        con.close(); audit("LOGIN_BLOCKED", target_id=row["id"], detail=f"Account status: {row['status']}")
        return jsonify({"error":"This account is disabled. Contact an administrator."}),403
    if not verify_password(password,row["password_hash"]):
        attempts=int(row["failed_attempts"])+1; locked=None
        if attempts>=LOGIN_MAX_FAILURES: locked=str(now+LOGIN_LOCK_MINUTES*60)
        con.execute("UPDATE users SET failed_attempts=?,locked_until=? WHERE id=?",(attempts,locked,row["id"])); con.commit(); con.close()
        audit("LOGIN_FAILURE", target_id=row["id"], detail=f"Failed attempt {attempts}")
        if locked: return jsonify({"error":"Too many failed attempts. Your account is locked for 15 minutes. An administrator can unlock it."}),423
        return jsonify({"error":"Invalid email or password."}),401
    # Rotate session and clear stale session data after successful authentication.
    session.clear(); session.permanent=True; session["user_id"]=row["id"]; session["session_version"]=row["session_version"]; session["csrf_token"]=secrets.token_urlsafe(32)
    con.execute("UPDATE users SET failed_attempts=0,locked_until=NULL,last_login_at=datetime('now','localtime') WHERE id=?",(row["id"],)); con.commit()
    fresh=con.execute("SELECT * FROM users WHERE id=?",(row["id"],)).fetchone(); con.close(); audit("LOGIN_SUCCESS", target_id=row["id"])
    return jsonify({"ok":True,"user":public_user(dict(fresh)),"csrf":session["csrf_token"]})

@app.post("/api/auth/register")
@csrf_required
def register():
    data=request.get_json(silent=True) or {}; name=str(data.get("name","")).strip(); email=str(data.get("email","")).strip().lower(); password=str(data.get("password",""))
    ok,msg=password_policy(password)
    if len(name)<2 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+",email) or not ok: return jsonify({"error":msg if not ok else "Enter a valid name and email."}),400
    con=get_db()
    if con.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
        con.close(); return jsonify({"error":"Public account creation is disabled after workspace initialization. Ask an administrator to create your account."}),403
    try:
        create_user(con,name,email,password,"Administrator"); con.commit(); row=con.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone()
    except Exception as e:
        if is_integrity_error(e):
            con.close(); return jsonify({"error":"An account with this email already exists."}),409
        raise
    except ValueError as e: con.close(); return jsonify({"error":str(e)}),400
    con.close(); session.clear(); session.permanent=True; session["user_id"]=row["id"]; session["session_version"]=row["session_version"]; session["csrf_token"]=secrets.token_urlsafe(32); audit("ACCOUNT_REGISTERED", target_id=row["id"])
    return jsonify({"ok":True,"user":public_user(dict(row)),"csrf":session["csrf_token"]})

@app.post("/api/auth/logout")
@login_required
@csrf_required
def logout():
    u=current_user(); audit("LOGOUT",actor_id=u["id"]); session.clear(); return jsonify({"ok":True})

@app.post("/api/auth/change-password")
@login_required
@csrf_required
def change_password():
    u=current_user(); data=request.get_json(silent=True) or {}; old=str(data.get("current_password","")); new=str(data.get("new_password",""))
    if not verify_password(old,u["password_hash"]): return jsonify({"error":"Current password is incorrect."}),400
    ok,msg=password_policy(new)
    if not ok: return jsonify({"error":msg}),400
    con=get_db(); new_version=int(u["session_version"])+1; con.execute("UPDATE users SET password_hash=?,password_changed_at=datetime('now','localtime'),session_version=? WHERE id=?",(hash_password(new),new_version,u["id"])); con.commit(); con.close(); audit("PASSWORD_CHANGED",actor_id=u["id"],target_id=u["id"])
    session.clear(); return jsonify({"ok":True,"message":"Password changed. Please sign in again."})

@app.get("/api/users")
@admin_required
def users():
    con=get_db(); rows=con.execute("SELECT id,name,email,role,status,created_at,last_login_at,failed_attempts,locked_until,mfa_enabled FROM users ORDER BY id DESC").fetchall(); con.close()
    return jsonify({"users":[dict(r) for r in rows]})

@app.post("/api/users")
@admin_required
@csrf_required
def create_workspace_user():
    actor=current_user(); data=request.get_json(silent=True) or {}; name=str(data.get("name","")).strip(); email=str(data.get("email","")).strip().lower(); password=str(data.get("password","")); role=str(data.get("role","Analyst")).strip()
    ok,msg=password_policy(password)
    if len(name)<2 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+",email) or not ok: return jsonify({"error":msg if not ok else "Enter a valid name and email."}),400
    if role not in ("Analyst","Administrator"): role="Analyst"
    con=get_db()
    try: create_user(con,name,email,password,role); con.commit(); row=con.execute("SELECT id,name,email,role,status,created_at,last_login_at,failed_attempts,locked_until,mfa_enabled FROM users WHERE email=?",(email,)).fetchone()
    except Exception as e:
        if is_integrity_error(e):
            con.close(); return jsonify({"error":"An account with this email already exists."}),409
        raise
    except ValueError as e: con.close(); return jsonify({"error":str(e)}),400
    con.close(); audit("USER_CREATED",actor_id=actor["id"],target_id=row["id"],detail=role); return jsonify({"ok":True,"user":dict(row)})

@app.patch("/api/users/<int:user_id>")
@admin_required
@csrf_required
def admin_update_user(user_id):
    actor=current_user(); data=request.get_json(silent=True) or {}; con=get_db(); target=con.execute("SELECT * FROM users WHERE id=?",(user_id,)).fetchone()
    if not target: con.close(); return jsonify({"error":"User not found."}),404
    if user_id==actor["id"] and data.get("status") in ("Disabled","Suspended"): con.close(); return jsonify({"error":"You cannot disable your own administrator account."}),400
    fields=[]; vals=[]; detail=[]
    if "role" in data and data["role"] in ("Analyst","Administrator"):
        # Never allow the last administrator to be demoted.
        if target["role"]=="Administrator" and data["role"]!="Administrator" and con.execute("SELECT COUNT(*) FROM users WHERE role='Administrator' AND status='Active'").fetchone()[0] <= 1:
            con.close(); return jsonify({"error":"At least one active administrator must remain."}),400
        fields.append("role=?"); vals.append(data["role"]); detail.append(f"role={data['role']}")
    if "status" in data and data["status"] in ("Active","Disabled"):
        if target["role"]=="Administrator" and data["status"]=="Disabled" and con.execute("SELECT COUNT(*) FROM users WHERE role='Administrator' AND status='Active'").fetchone()[0] <= 1:
            con.close(); return jsonify({"error":"The last active administrator cannot be disabled."}),400
        fields.append("status=?"); vals.append(data["status"]); detail.append(f"status={data['status']}")
    if data.get("unlock"):
        fields += ["failed_attempts=?","locked_until=?"]; vals += [0,None]; detail.append("unlocked")
    if fields:
        vals.append(user_id); con.execute(f"UPDATE users SET {', '.join(fields)} WHERE id=?",vals)
    if data.get("revoke_sessions"):
        con.execute("UPDATE users SET session_version=session_version+1 WHERE id=?",(user_id,)); detail.append("sessions revoked")
    con.commit(); row=con.execute("SELECT id,name,email,role,status,created_at,last_login_at,failed_attempts,locked_until,mfa_enabled FROM users WHERE id=?",(user_id,)).fetchone(); con.close(); audit("USER_UPDATED",actor_id=actor["id"],target_id=user_id,detail=", ".join(detail)); return jsonify({"ok":True,"user":dict(row)})

@app.post("/api/users/<int:user_id>/reset-password")
@admin_required
@csrf_required
def admin_reset_password(user_id):
    actor=current_user(); data=request.get_json(silent=True) or {}; new=str(data.get("password","")); ok,msg=password_policy(new)
    if not ok: return jsonify({"error":msg}),400
    con=get_db(); target=con.execute("SELECT id FROM users WHERE id=?",(user_id,)).fetchone()
    if not target: con.close(); return jsonify({"error":"User not found."}),404
    con.execute("UPDATE users SET password_hash=?,password_changed_at=datetime('now','localtime'),failed_attempts=0,locked_until=NULL,session_version=session_version+1 WHERE id=?",(hash_password(new),user_id)); con.commit(); con.close(); audit("ADMIN_PASSWORD_RESET",actor_id=actor["id"],target_id=user_id); return jsonify({"ok":True,"message":"Password reset and active sessions revoked."})

@app.get("/api/audit")
@admin_required
def audit_events():
    con=get_db(); rows=con.execute("""SELECT a.id,a.event,a.detail,a.ip,a.created_at,u.name AS actor,t.name AS target FROM audit_log a LEFT JOIN users u ON u.id=a.actor_user_id LEFT JOIN users t ON t.id=a.target_user_id ORDER BY a.id DESC LIMIT 100""").fetchall(); con.close(); return jsonify({"events":[dict(r) for r in rows]})

@app.get("/api/dashboard")
@login_required
def dashboard():
    m=load_metrics(); u=current_user(); con=get_db(); total=con.execute("SELECT COUNT(*) FROM scans").fetchone()[0]; phishing=con.execute("SELECT COUNT(*) FROM scans WHERE verdict='PHISHING'").fetchone()[0]; safe=total-phishing
    recent=[dict(r) for r in con.execute("SELECT id,url,verdict,risk,confidence,model,created_at FROM scans ORDER BY id DESC LIMIT 8").fetchall()]; con.close()
    return jsonify({"user":public_user(u),"dataset":m.get("dataset",{}),"models":m.get("models",{}),"best_model":m.get("best_model"),"scan_stats":{"total":total,"phishing":phishing,"safe":safe,"open_incidents":phishing},"recent":recent})

@app.post("/api/predict")
@login_required
@csrf_required
def predict():
    data=request.get_json(silent=True) or {}; raw=str(data.get("url","")).strip()
    if not raw or len(raw)>4096: return jsonify({"error":"Enter a valid URL (maximum 4096 characters)."}),400
    if not BUNDLE_PATH.exists(): return jsonify({"error":"ML models are not deployed."}),503
    try:
        f=extract_url_features(raw); bundle=load_bundle(); metrics=load_metrics(); x=f["values"]; model_name=metrics.get("best_model") or "Random Forest"
        if model_name not in bundle.get("models",{}): model_name="Random Forest"
        p=max(0.0,min(1.0,predict_probability(bundle,model_name,x))); phishing=p>=0.5; confidence=(p if phishing else 1-p)*100; risk=p*100; verdict="PHISHING" if phishing else "SAFE"; u=current_user()
        con=get_db(); cur=con.execute("INSERT INTO scans(user_id,url,verdict,risk,confidence,model,features_json,created_at) VALUES(?,?,?,?,?,?,?,datetime('now','localtime')) RETURNING id",(u["id"],f["url"],verdict,risk,confidence,model_name,json.dumps(f))); scan_id=cur.fetchone()[0]; con.commit(); con.close()
        if phishing: audit("PHISHING_DETECTED",actor_id=u["id"],detail=f"scan_id={scan_id}")
        return jsonify({"id":scan_id,"prediction":verdict,"is_phishing":phishing,"confidence":round(confidence,2),"risk":round(risk,2),"phishing_probability":round(p*100,2),"features":f,"model":model_name,"model_accuracy":metrics.get("models",{}).get(model_name,{}).get("accuracy")})
    except Exception as e: return jsonify({"error":str(e)}),400

@app.get("/api/history")
@login_required
def history():
    con=get_db(); rows=con.execute("SELECT id,url,verdict,risk,confidence,model,created_at FROM scans WHERE user_id=? ORDER BY id DESC LIMIT 100",(current_user()["id"],)).fetchall(); con.close(); return jsonify({"scans":[dict(r) for r in rows]})

@app.delete("/api/history")
@login_required
@csrf_required
def clear_history():
    u=current_user(); con=get_db(); con.execute("DELETE FROM scans WHERE user_id=?",(u["id"],)); con.commit(); con.close(); audit("HISTORY_CLEARED",actor_id=u["id"]); return jsonify({"ok":True})

@app.get("/api/incidents")
@login_required
def incidents():
    con=get_db(); rows=con.execute("SELECT id,url,risk,confidence,model,created_at FROM scans WHERE verdict='PHISHING' ORDER BY id DESC LIMIT 100").fetchall(); con.close(); out=[]
    for r in rows:
        d=dict(r); d["severity"]="CRITICAL" if d["risk"]>=85 else "HIGH" if d["risk"]>=65 else "MEDIUM"; d["status"]="Active"; out.append(d)
    return jsonify({"incidents":out})

@app.get("/api/scan/<int:scan_id>")
@login_required
def scan_detail(scan_id):
    con=get_db(); row=con.execute("SELECT * FROM scans WHERE id=? AND user_id=?",(scan_id,current_user()["id"])).fetchone(); con.close()
    if not row: return jsonify({"error":"Scan not found"}),404
    d=dict(row); d["features"]=json.loads(d.pop("features_json")); return jsonify(d)

@app.get("/<path:path>")
def static_files(path):
    target=FRONTEND/path
    if target.exists() and target.is_file():
        response = send_from_directory(FRONTEND,path)
        # Never let browser/CDN caches keep an obsolete frontend during local development.
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response
    response = send_from_directory(FRONTEND,"index.html")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response

init_db(ROOT)
if __name__=="__main__": app.run(host="127.0.0.1",port=5000,debug=False)
