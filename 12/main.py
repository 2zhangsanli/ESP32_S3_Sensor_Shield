# ESP32-S3 HAR - WiFi AP + 动态网页
import gc, math, time, _thread, network, socket
time.sleep(3)
from machine import SoftI2C, Pin, reset
import neopixel
from array import array
try:
    import ujson as json
except:
    import json

SDA_PIN=8;SCL_PIN=9;LED_PIN=48;MPU_ADDR=0x68
WINDOW_SIZE=128;WINDOW_STEP=64;SAMPLE_MS=20;N_AXES=6;N_FEATURES=30;SMOOTH_FRAMES=2
WIFI_SSID="vivo"
WIFI_PW="12345678"

ACT_NAMES=["sit","stand","walk","upstairs","downstairs","run"]
LED_C={0:(255,0,0),1:(0,255,0),2:(0,0,255),3:(255,255,0),4:(255,0,255),5:(255,255,255)}
ICONS=["&#x1F4BA;","&#x1F9CD;","&#x1F6B6;","&#x1F51D;","&#x1F51F;","&#x1F3C3;"]
CN=["静坐","站立","行走","上楼","下楼","跑步"]
CLR=["#ff4444","#44ff44","#4488ff","#ffff44","#ff44ff","#ff8844"]

current_action="init";current_votes=0;inference_ready=False;page_count=0
np=None;i2c=None

def init_led():
    global np
    if np is None:np=neopixel.NeoPixel(Pin(LED_PIN),1)
def set_led(r,g,b):
    init_led()
    if np:np[0]=(r,g,b);np.write()
def init_i2c():
    global i2c
    if i2c is None:i2c=SoftI2C(sda=Pin(SDA_PIN),scl=Pin(SCL_PIN),freq=100000)

def conv(v):return v-65536 if v>=32768 else v
def init_mpu():
    init_i2c()
    for _ in range(5):
        try:i2c.writeto_mem(MPU_ADDR,0x6B,b'\x00');time.sleep_ms(50);return True
        except:time.sleep_ms(50)
    return False
def read_mpu_raw():
    for _ in range(5):
        try:
            buf=i2c.readfrom_mem(MPU_ADDR,0x3B,14)
            ax=conv((buf[0]<<8)|buf[1]);ay=conv((buf[2]<<8)|buf[3])
            az=conv((buf[4]<<8)|buf[5]);gx=conv((buf[8]<<8)|buf[9])
            gy=conv((buf[10]<<8)|buf[11]);gz=conv((buf[12]<<8)|buf[13])
            if abs(ax)>32000 or abs(ay)>32000 or abs(az)>32000:continue
            return ax,ay,az,gx,gy,gz
        except:time.sleep_ms(10)
    return None

ax_off=ay_off=az_off=gx_off=gy_off=gz_off=0
def calibrate(n=100):
    global ax_off,ay_off,az_off,gx_off,gy_off,gz_off
    s=[0]*6;v=0;t0=time.ticks_ms()
    while v<n and time.ticks_diff(time.ticks_ms(),t0)<10000:
        d=read_mpu_raw()
        if d is None:time.sleep_ms(10);continue
        for i in range(6):s[i]+=d[i]
        v+=1;time.sleep_ms(10)
    if v>0:ax_off,ay_off,az_off,gx_off,gy_off,gz_off=[x//v for x in s]

def get_cal():
    d=read_mpu_raw()
    if d is None:return None
    return (d[0]-ax_off,d[1]-ay_off,d[2]-az_off,d[3]-gx_off,d[4]-gy_off,d[5]-gz_off)

def extract_features(buf):
    feats=array('f',[0.0]*N_FEATURES)
    for axis in range(N_AXES):
        s=0.0;mn=1e9;mx=-1e9
        for i in range(WINDOW_SIZE):
            v=buf[i*N_AXES+axis];s+=v
            if v<mn:mn=v
            if v>mx:mx=v
        mean=s/WINDOW_SIZE;sq=0.0
        for i in range(WINDOW_SIZE):
            d=buf[i*N_AXES+axis]-mean;sq+=d*d
        b=axis*5
        feats[b]=mean;feats[b+1]=math.sqrt(sq/WINDOW_SIZE)
        feats[b+2]=mx;feats[b+3]=mn;feats[b+4]=mx-mn
    return feats

def rf_predict(feats,trees,sm,ss,nc):
    fs=array('f',[0.0]*N_FEATURES)
    for i in range(N_FEATURES):fs[i]=(feats[i]-sm[i])/ss[i]
    votes=[0]*nc
    for tree in trees:
        node=0
        fa=tree["feature"];ta=tree["threshold"]
        la=tree["children_left"];ra=tree["children_right"]
        va=tree["value"]
        while fa[node]!=-2:
            if fs[fa[node]]<=ta[node]:node=la[node]
            else:node=ra[node]
        lv=va[node][0];bc=0;bv=lv[0]
        for c in range(1,len(lv)):
            if lv[c]>bv:bv=lv[c];bc=c
        votes[bc]+=1
    w=0
    for c in range(1,nc):
        if votes[c]>votes[w]:w=c
    return w,votes[w]

def build_page(action_name,votes,pcnt):
    try:idx=ACT_NAMES.index(action_name)
    except:idx=-1
    if idx>=0:icon=ICONS[idx];cn=CN[idx];color=CLR[idx]
    else:icon="&#x23F3;";cn=action_name;color="#888"
    pct=str(min(100,int(votes*100/15)))+"%"
    return """<!DOCTYPE html><html><head>
<meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1.0,user-scalable=no'>
<title>HAR Detection</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0d0d1a;color:#fff;display:flex;
align-items:center;justify-content:center;min-height:100vh;text-align:center;padding:20px}
.card{border:3px solid #888;border-radius:28px;padding:50px 30px;max-width:380px;width:100%;
box-shadow:0 0 60px rgba(255,255,255,0.05);transition:all .4s}
.icon{font-size:5em;line-height:1.3}
.name{font-size:2.8em;font-weight:900;color:#888;margin:18px 0 8px;letter-spacing:2px}
.info{color:#666;font-size:.95em}
.bar{margin-top:25px;height:6px;background:#222;border-radius:3px;overflow:hidden}
.bar-inner{height:100%;background:#888;width:0%;border-radius:3px;transition:width .5s}
.footer{margin-top:35px;color:#444;font-size:.7em;letter-spacing:1px}
</style></head><body>
<div class='card' id='card'>
<div class='icon' id='icon'>Loading...</div>
<div class='name' id='name'>--</div>
<div class='info' id='info'></div>
<div class='bar'><div class='bar-inner' id='bar'></div></div>
<div class='footer' id='footer'></div>
</div>
<script>
var icons={sit:'&#x1F4BA;',stand:'&#x1F9CD;',walk:'&#x1F6B6;',upstairs:'&#x1F51D;',downstairs:'&#x1F51F;',run:'&#x1F3C3;'};
var cn={sit:'静坐',stand:'站立',walk:'行走',upstairs:'上楼',downstairs:'下楼',run:'跑步'};
var clr={sit:'#ff4444',stand:'#44ff44',walk:'#4488ff',upstairs:'#ffff44',downstairs:'#ff44ff',run:'#ff8844'};
var pc=0;
function update(){
 fetch('/api?_='+(++pc)).then(function(r){return r.json()}).then(function(d){
  var a=d.a; var v=parseInt(d.v)||0;
  if(a==='init') return;
  var pct=Math.min(100,Math.round(v/15*100));
  document.getElementById('icon').innerHTML=icons[a]||'?';
  document.getElementById('name').textContent=cn[a]||a;
  document.getElementById('name').style.color=clr[a]||'#888';
  document.getElementById('card').style.borderColor=clr[a]||'#888';
  document.getElementById('card').style.boxShadow='0 0 60px '+(clr[a]||'#888')+'22';
  document.getElementById('info').textContent='Confidence: '+pct+'% | Votes: '+v+'/15';
  document.getElementById('bar').style.background=clr[a]||'#888';
  document.getElementById('bar').style.width=pct+'%';
  document.getElementById('footer').textContent='Refresh #'+pc+' | ESP32-S3';
 }).catch(function(){});
}
setInterval(update,500);
update();
</script></body></html>"""

def inference_loop():
    global current_action,current_votes,inference_ready
    set_led(0,0,255);time.sleep(0.5)
    gc.collect()
    with open("rf_params.json","r") as f:params=json.load(f)
    gc.collect()
    trees=params["trees"];sm=params["scaler_mean"];ss=params["scaler_scale"];nc=params["n_classes"]
    for _ in range(5):set_led(255,255,0);time.sleep(0.1);set_led(0,0,0);time.sleep(0.1)
    calibrate(100)
    BUF_SZ=WINDOW_SIZE*N_AXES;buf=array('f',[0.0]*BUF_SZ);pos=0
    sq=array('b',[0]*SMOOTH_FRAMES);sqp=0;last_t=time.ticks_ms()
    inference_ready=True;set_led(0,10,0)
    while True:
        now=time.ticks_ms()
        if time.ticks_diff(now,last_t)<SAMPLE_MS:time.sleep_ms(2);continue
        d=get_cal()
        if d is None:time.sleep_ms(5);continue
        last_t=now
        for v in d:buf[pos]=float(v);pos+=1
        if pos<BUF_SZ:continue
        feats=extract_features(buf)
        try:pred,votes=rf_predict(feats,trees,sm,ss,nc)
        except:pred=-1;votes=0
        sq[sqp]=pred;sqp=(sqp+1)%SMOOTH_FRAMES
        same=True
        for i in range(1,SMOOTH_FRAMES):
            if sq[i]!=sq[0]:same=False;break
        if same and pred>=0:
            current_action=ACT_NAMES[pred];current_votes=votes
            set_led(*LED_C.get(pred,(255,100,0)))
        shift=WINDOW_STEP*N_AXES;keep=BUF_SZ-shift
        for i in range(keep):buf[i]=buf[i+shift]
        pos=keep

def main():
    global page_count
    set_led(255,100,0)
    if not init_mpu():
        set_led(255,0,0)
        while True:time.sleep(1)
    gc.collect()
    ap=network.WLAN(network.AP_IF);ap.active(False)
    sta=network.WLAN(network.STA_IF);sta.active(False)
    time.sleep(0.5)

    # STA模式连接热点
    sta.active(True);time.sleep(0.3)
    sta.connect(WIFI_SSID,WIFI_PW)

    for _ in range(30):
        if sta.isconnected():break
        set_led(0,0,255);time.sleep_ms(300)
        set_led(0,0,0);time.sleep_ms(200)

    if sta.isconnected():
        set_led(0,255,0)
        web_ip=sta.ifconfig()[0]
    else:
        set_led(255,0,0)
        web_ip=""
        # WiFi失败: 红灯闪, 5秒后重启
        for _ in range(6):set_led(255,0,0);time.sleep(0.3);set_led(0,0,0);time.sleep(0.3)
        reset()
    _thread.start_new_thread(inference_loop,())
    for _ in range(50):
        if inference_ready:break
        time.sleep_ms(100)
    addr=socket.getaddrinfo('0.0.0.0',80)[0][-1]
    s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind(addr);s.listen(5);s.settimeout(1)
    req_cnt=0
    while True:
        try:cl,addr2=s.accept()
        except OSError:continue
        except Exception as e:
            try:
                with open("err.log","w") as f:f.write(f"accept:{e}")
            except:pass
            continue
        try:
            cl.settimeout(2)
            req=cl.recv(512).decode('utf-8','ignore')
            req_cnt+=1
            if '/api' in req or 'api' in req.lower():
                resp='{"a":"%s","v":%d}'%(current_action.replace('"','\\"'),current_votes)
                cl.send('HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nCache-Control: no-cache, no-store, must-revalidate\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n')
                cl.send(resp)
            else:
                page_count+=1
                page=build_page(current_action,current_votes,page_count)
                cl.send('HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-cache, no-store, must-revalidate\r\nConnection: close\r\n\r\n')
                cl.send(page)
            # 每10个请求记录一次
            if req_cnt%10==0:
                try:
                    with open("req.log","w") as f:f.write(str(req_cnt))
                except:pass
        except Exception as e:
            try:
                with open("err.log","w") as f:f.write(f"send:{e}")
            except:pass
        finally:
            try:cl.close()
            except:pass

main()
