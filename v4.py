# SATB Arranger - Version 3.2
# 保留钢琴卷帘；把 力度(dyn/vel) 与 延音(ped) 写到 SATB 文本音符属性中，并与卷帘互通
# 语法示例： C4q{dyn=mf}  A3h{vel=110}  Rq{ped=on}  C4e{ped=off,vel=85}
# 导出 MIDI 写入 CC64 踏板事件；播放时三后端均支持 CC64

import os, sys, re, time, threading, tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pygame
import pygame.midi
import pygame.sndarray
from midiutil import MIDIFile

# ---------- Windows: FluidSynth DLL 路径 ----------
if sys.platform == "win32":
    for _cand in [os.getenv("FLUIDSYNTH_BIN"), r"C:\tools\fluidsynth\bin"]:
        if _cand and os.path.isdir(_cand):
            try:
                os.add_dll_directory(_cand)
                break
            except Exception:
                pass

# ---------- 可选导入 FluidSynth ----------
HAVE_FS = False
try:
    import fluidsynth
    HAVE_FS = True
except Exception:
    HAVE_FS = False

APP_TITLE   = "SATB Arranger — v3.2 (Roll + Note Attributes for dyn/vel/ped)"
DEFAULT_BPM = 90
DEFAULT_METER = (4, 4)
GRID_DEFAULT = 0.25  # 1/16 拍

NOTE_MAP = {
    "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,
    "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11
}
DUR_MAP = {"w":4.0,"h":2.0,"q":1.0,"e":0.5,"s":0.25,"t":0.125}

VOICE_ORDER = ["S","A","T","B"]
VOICE_TO_CH = {v:i for i,v in enumerate(VOICE_ORDER)}
VOICE_COLOR = {"S":"#f26b6b","A":"#f2a35e","T":"#59c972","B":"#5aa6f2"}
RANGES = {"S":(60,84),"A":(55,77),"T":(48,69),"B":(40,60)}

# 强弱 <-> 力度 映射
DYN_TO_VEL = {"pp":30,"p":50,"mp":70,"mf":85,"f":100,"ff":115}
VEL_TO_DYN = [(30,"pp"),(50,"p"),(70,"mp"),(85,"mf"),(100,"f"),(115,"ff")]

def vel_to_dyn_or_num(vel: int):
    # 找到最近的强弱标记；若相差<=6则用 dyn，否则写 vel=数值
    best = min(VEL_TO_DYN, key=lambda kv: abs(kv[0]-vel))
    if abs(best[0]-vel) <= 6:
        return ("dyn", best[1])
    return ("vel", vel)

def clamp(v, lo, hi): return max(lo, min(hi, v))
def beats_to_seconds(beats, bpm): return (60.0/bpm)*beats
def q(x, grid=GRID_DEFAULT): return round(x / grid) * grid

def note_to_midi(token: str):
    if token.upper().startswith("R"): return None
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)", token.strip())
    if not m: raise ValueError(f"音名错误: {token}")
    name = m.group(1).upper()+m.group(2); octv = int(m.group(3))
    if name not in NOTE_MAP: raise ValueError(f"不认识的音名: {name}")
    return 12*(octv+1)+NOTE_MAP[name]

# ----------- 扩展解析：音符(+时值) + 可选属性块 {key=val,...} -----------
# 支持：vel=1..127  dyn=pp|p|mp|mf|f|ff  ped=on|off
def parse_token(tok: str):
    tok = tok.strip()
    if not tok: return None
    # 拆属性块
    attr = {}
    attr_m = re.search(r"\{([^}]*)\}\s*$", tok)
    if attr_m:
        body = attr_m.group(1).strip()
        tok  = tok[:attr_m.start()].strip()
        if body:
            for pair in body.split(","):
                if not pair.strip(): continue
                if "=" in pair:
                    k,v = pair.strip().split("=",1)
                    k = k.strip().lower(); v = v.strip().lower()
                    attr[k]=v
                else:
                    # 允许单词形式，如 {ped} -> ped=on
                    if pair.strip().lower()=="ped": attr["ped"]="on"

    # 休止
    if tok[0].upper()=="R":
        m = re.fullmatch(r"R([whqest])(\.*)?", tok, re.IGNORECASE)
        if m:
            base = DUR_MAP[m.group(1).lower()]
            dots = (m.group(2) or "")
            return (None, base*(1.5**len(dots)), attr)
        m2 = re.fullmatch(r"R(\d+(?:\.\d+)?)", tok, re.IGNORECASE)
        if m2: return (None, float(m2.group(1)), attr)
        raise ValueError(f"休止符格式错误: {tok}")

    # 音符 + 记号时值
    m = re.fullmatch(r"([A-Ga-g][#b]?[-]?\d)([whqest])(\.*)?", tok)
    if m:
        pitch = note_to_midi(m.group(1))
        base = DUR_MAP[m.group(2).lower()]
        dots = (m.group(3) or "")
        return (pitch, base*(1.5**len(dots)), attr)

    # 音符 + 数值时值
    m2 = re.fullmatch(r"([A-Ga-g][#b]?[-]?\d)(\d+(?:\.\d+)?)", tok)
    if m2:
        return (note_to_midi(m2.group(1)), float(m2.group(2)), attr)

    raise ValueError(f"音符格式错误: {tok}")

def dur_to_token(d):
    for k,v in DUR_MAP.items():
        if abs(d - v) < 1e-6: return k
    s=f"{d:.2f}".rstrip("0").rstrip("."); return s if s else "0"

def format_pitch(p):
    pc = p%12; octv=p//12-1
    for k,v in NOTE_MAP.items():
        if v==pc and len(k)<=2: return f"{k}{octv}"
    return f"C{octv}"

# ====================== 数据模型（OOP） ======================
class Note:
    __slots__=("pitch","beats","vel")
    def __init__(self, pitch, beats, vel=96):
        self.pitch=pitch; self.beats=beats; self.vel=vel

class Part:
    def __init__(self, name): self.name=name; self.notes=[]
    def add(self, n:Note): self.notes.append(n)
    def total_beats(self): return sum(n.beats for n in self.notes)
    def extend(self, seq): self.notes.extend(seq)
    def with_starts(self):
        t=0.0; out=[]
        for n in self.notes:
            out.append((t,n)); t+=n.beats
        return out

class Score:
    def __init__(self, bpm=DEFAULT_BPM, meter=DEFAULT_METER):
        self.bpm=bpm; self.meter=meter
        self.parts={v:Part(v) for v in VOICE_ORDER}
        self.sustain=[]   # [(start, end)]
    def ensure_aligned(self):
        mx=max(self.parts[v].total_beats() for v in VOICE_ORDER)
        for v in VOICE_ORDER:
            t=self.parts[v].total_beats()
            if t<mx: self.parts[v].add(Note(None, mx-t))
    def total_beats(self):
        self.ensure_aligned()
        return max(self.parts[v].total_beats() for v in VOICE_ORDER)

# ====================== 配器（和弦→四声部） ======================
CHORD_QUAL = {
    "":[0,4,7],"m":[0,3,7],"dim":[0,3,6],"aug":[0,4,8],
    "6":[0,4,7,9],"7":[0,4,7,10],"maj7":[0,4,7,11],
    "m7":[0,3,7,10],"m6":[0,3,7,9],"m7b5":[0,3,6,10]
}
def parse_chord(ch):
    m=re.fullmatch(r"([A-Ga-g])([#b]?)([a-zA-Z0-9#b]*)", ch.strip())
    if not m: raise ValueError(f"和弦错误: {ch}")
    root=m.group(1).upper()+m.group(2); qual=(m.group(3) or "").lower()
    rep={"maj7":"maj7","maj":"maj","min":"m","min7":"m7","m7b5":"m7b5","dim7":"dim7"}
    if qual in rep: qual=rep[qual]
    if qual=="dim7": pcs=[0,3,6,9]
    else:
        if qual not in CHORD_QUAL:
            q2=qual[:3]; qual=q2 if q2 in CHORD_QUAL else (qual[:1] if qual[:1] in CHORD_QUAL else "")
        pcs=CHORD_QUAL[qual][:]
    root_pc=NOTE_MAP[root]; return root_pc, [(p+root_pc)%12 for p in pcs]

def nearest_pitch(prev, pc, lo, hi):
    if prev is None:
        mid=(lo+hi)//2; cand=pc
        while cand<lo: cand+=12
        while cand>hi: cand-=12
        while cand+12<=hi and abs(cand+12-mid)<abs(cand-mid): cand+=12
        while cand-12>=lo and abs(cand-12-mid)<abs(cand-mid): cand-=12
        return cand
    base=(prev//12)*12+pc
    cands=[clamp(base-12,lo,hi),clamp(base,lo,hi),clamp(base+12,lo,hi)]
    cands.sort(key=lambda x:abs(x-prev)); return cands[0]

class Arranger:
    @staticmethod
    def chord_prog_to_satb(chords, durs, last=None):
        beats=[]
        for d in durs:
            d=d.strip().lower()
            beats.append(DUR_MAP[d] if d in DUR_MAP else float(d))
        if len(beats)!=len(chords): raise ValueError("和弦与时值数量不一致")
        prev = last or {"S":None,"A":None,"T":None,"B":None}
        parts={v:[] for v in VOICE_ORDER}
        for ch,b in zip(chords,beats):
            _, pcs=parse_chord(ch)
            if len(pcs)==3: pcs=pcs+[pcs[0]]
            b_pc,s_pc,a_pc,t_pc = pcs[0], pcs[1%len(pcs)], pcs[2%len(pcs)], pcs[3%len(pcs)]
            bp=nearest_pitch(prev["B"], b_pc, *RANGES["B"])
            tp=nearest_pitch(prev["T"], t_pc, *RANGES["T"])
            ap=nearest_pitch(prev["A"], a_pc, *RANGES["A"])
            sp=nearest_pitch(prev["S"], s_pc, *RANGES["S"])
            parts["B"].append(Note(bp,b)); parts["T"].append(Note(tp,b))
            parts["A"].append(Note(ap,b)); parts["S"].append(Note(sp,b))
            prev.update({"B":bp,"T":tp,"A":ap,"S":sp})
        return parts

# ====================== 事件渲染（含 CC64） ======================
def score_to_events(score: Score, transpose=0):
    ev=[]
    for v in VOICE_ORDER:
        ch=VOICE_TO_CH[v]; t=0.0
        for n in score.parts[v].notes:
            sec=beats_to_seconds(n.beats, score.bpm)
            if n.pitch is not None:
                p=n.pitch+transpose
                ev.append((t, True,  p, ch, n.vel))
                ev.append((t+sec*0.98, False, p, ch, 0))
            t+=sec
    # Pedal CC64 按区段生成开/关
    for (s,e) in score.sustain:
        ts=beats_to_seconds(s, score.bpm); te=beats_to_seconds(e, score.bpm)
        for ch in range(4):
            ev.append((ts, True, -1, ch, 127))
            ev.append((te, True, -1, ch,   0))
    ev.sort(key=lambda e:(e[0], 0 if not e[1] else 1))
    return ev

# ====================== 规则检查（略同 v3.1） ======================
class RuleChecker:
    def __init__(self, sc:Score): self.sc=sc; self.issues=[]
    def check(self):
        self.issues.clear(); self._range(); self._cross(); self._parallels(); return self.issues
    def _range(self):
        for v in VOICE_ORDER:
            lo,hi=RANGES[v]; t=0.0
            for n in self.sc.parts[v].notes:
                if n.pitch is not None and not (lo<=n.pitch<=hi):
                    self.issues.append({"beat":t,"type":"越界","detail":f"{v} {format_pitch(n.pitch)} 不在[{lo}-{hi}]","voice":v})
                t+=n.beats
    def _timeline(self):
        pts={0.0}
        for v in VOICE_ORDER:
            t=0.0
            for n in self.sc.parts[v].notes:
                pts.add(t); t+=n.beats; pts.add(t)
        pts=sorted(pts); out=[]
        for b in pts:
            snap={}
            for v in VOICE_ORDER:
                t=0.0; cur=None
                for n in self.sc.parts[v].notes:
                    if t<=b<t+n.beats: cur=n.pitch; break
                    t+=n.beats
                snap[v]=cur
            out.append((b,snap))
        return out
    def _cross(self):
        tl=self._timeline()
        for b,s in tl:
            s_,a_,t_,b_ = s["S"],s["A"],s["T"],s["B"]
            if None not in (s_,a_) and s_ < a_:
                self.issues.append({"beat":b,"type":"交叉","detail":"S < A","voice":"S/A"})
            if None not in (a_,t_) and a_ < t_:
                self.issues.append({"beat":b,"type":"交叉","detail":"A < T","voice":"A/T"})
            if None not in (t_,b_) and t_ < b_:
                self.issues.append({"beat":b,"type":"交叉","detail":"T < B","voice":"T/B"})
    def _parallels(self):
        tl=self._timeline(); pairs=[("S","A"),("A","T"),("T","B")]
        for i in range(1,len(tl)):
            b0,s0=tl[i-1]; b1,s1=tl[i]
            for (u,v) in pairs:
                p0u,p0v=s0[u],s0[v]; p1u,p1v=s1[u],s1[v]
                if None in (p0u,p0v,p1u,p1v): continue
                int0=abs(p0u-p0v); int1=abs(p1u-p1v)
                du=p1u-p0u; dv=p1v-p0v
                same=(du>0 and dv>0) or (du<0 and dv<0)
                moved=(du!=0 or dv!=0)
                if same and moved:
                    if int0 in (7,19) and int1 in (7,19):
                        self.issues.append({"beat":b1,"type":"并行五度","detail":f"{u}-{v} 在 {b0:.2f}→{b1:.2f}","voice":f"{u}/{v}"})
                    if int0%12==0 and int1%12==0:
                        self.issues.append({"beat":b1,"type":"并行八度","detail":f"{u}-{v} 在 {b0:.2f}→{b1:.2f}","voice":f"{u}/{v}"})

# ====================== Piano Roll（保留 v3.1 的交互） ======================
class PianoRollCanvas(tk.Canvas):
    def __init__(self, master, on_modified, get_active_voice, set_active_voice, get_grid, **kw):
        super().__init__(master, bg="#0f1115", highlightthickness=0, **kw)
        self.beat_px=40; self.pitch_px=6; self.margin_left=48; self.margin_top=10
        self.cursor=None; self.total_beats=0; self.min_pitch=36; self.max_pitch=84
        self._cursor_running=False; self._cursor_start_ts=0; self._bpm=DEFAULT_BPM
        self._scroll_follow=True; self._scroll_w=0; self._scroll_h=0
        self._score=None; self._sustain=[]
        self.on_modified=on_modified
        self.get_active_voice=get_active_voice
        self.set_active_voice=set_active_voice
        self.get_grid=get_grid
        self._hit=None; self._drag_pedal=None; self._selected=None; self._resize_grip=10
        self.xscroll=ttk.Scrollbar(master, orient="horizontal", command=self.xview)
        self.configure(xscrollcommand=self.xscroll.set); self.xscroll.pack(fill="x", side="bottom")
        self.bind("<Button-1>", self.on_left_down)
        self.bind("<B1-Motion>", self.on_left_drag)
        self.bind("<ButtonRelease-1>", self.on_left_up)
        self.bind("<Double-1>", self.on_double_left)
        self.bind("<Button-3>", self.on_right_click)
        self.bind("<Shift-B1-Motion>", self.on_shift_drag)
        self.bind("<Delete>", self.on_key_delete); self.focus_set()

    def set_score(self, sc, sustain=None):
        self.delete("all"); self._score=sc
        if sustain is not None: self._sustain=list(sustain)
        self._bpm=sc.bpm; self.total_beats=sc.total_beats()
        ps=[]; 
        for v in VOICE_ORDER:
            for t,n in sc.parts[v].with_starts():
                if n.pitch is not None: ps.append(n.pitch)
        if ps: self.min_pitch=max(24,min(ps)-3); self.max_pitch=min(96,max(ps)+3)
        else:  self.min_pitch,self.max_pitch=36,84
        w=self.margin_left+int(self.total_beats*self.beat_px)+200
        main_h=self.margin_top+(self.max_pitch-self.min_pitch+1)*self.pitch_px+20
        pedal_h=32; h=main_h+pedal_h+12
        self._scroll_w=w; self._scroll_h=h
        self.config(scrollregion=(0,0,w,h), width=min(1100,w), height=380)
        self._draw_grid(sc, main_h, pedal_h)
        self._draw_notes(sc)
        self._draw_sustain(main_h, pedal_h)
        x=self.margin_left
        if self.cursor: self.delete(self.cursor)
        self.cursor=self.create_line(x,self.margin_top,x,main_h-10, fill="#ffffaa", width=2)

    def beat_to_x(self,b): return self.margin_left + b*self.beat_px
    def x_to_beat(self,x): return max(0.0,(x-self.margin_left)/self.beat_px)
    def pitch_to_y(self,p): return self.margin_top + (self.max_pitch-p)*self.pitch_px
    def y_to_pitch(self,y): return clamp(self.max_pitch - int((y-self.margin_top)/self.pitch_px), self.min_pitch, self.max_pitch)

    def _draw_grid(self, sc, main_h, pedal_h):
        num,den=sc.meter; total=int(np.ceil(self.total_beats))
        for b in range(total+1):
            x=self.beat_to_x(b); col="#2a2f3a"; w=1
            if b%num==0: col="#3b4252"; w=2
            elif b%num==1: col="#333846"
            self.create_line(x, self.margin_top, x, main_h-10, fill=col, width=w)
        for p in range(self.min_pitch,self.max_pitch+1):
            y=self.pitch_to_y(p); col="#161a20" if p%12 else "#1b2028"
            self.create_line(self.margin_left,y,self._scroll_w,y, fill=col, width=1)
        for p in range(self.min_pitch,self.max_pitch+1):
            if p%12==0:
                y=self.pitch_to_y(p); self.create_text(22,y-6,text=f"C{p//12-1}", fill="#aaa", font=("Consolas", 9))
        y0=main_h+6
        self.create_rectangle(self.margin_left,y0,self._scroll_w,y0+pedal_h, fill="#101418", outline="#2a2f3a")
        self.create_text(30,y0+pedal_h/2, text="Ped", fill="#86c591", font=("Segoe UI", 9))

    def _draw_notes(self, sc):
        self._map={}
        for v in VOICE_ORDER:
            color=VOICE_COLOR[v]; t=0.0
            for idx,n in enumerate(sc.parts[v].notes):
                if n.pitch is not None and n.beats>0:
                    x0=self.beat_to_x(t); x1=self.beat_to_x(t+n.beats)
                    y0=self.pitch_to_y(n.pitch)+0.2; y1=y0+int(self.pitch_px*0.8)
                    rid=self.create_rectangle(x0,y0,x1,y1, fill=color, outline="", tags=("note", v))
                    self.create_rectangle(x1-10, y0, x1, y1, fill="#eaeaea", outline="", tags=("grip", v))
                    if self._selected == (v, idx):
                        self.create_rectangle(x0,y0,x1,y1, outline="#ffffaa", width=2)
                    self._map[rid]=(v,idx)
                t+=n.beats

    def _draw_sustain(self, main_h, pedal_h):
        self._ped_map=[]
        y0=main_h+6
        for i,(s,e) in enumerate(self._sustain):
            x0=self.beat_to_x(s); x1=self.beat_to_x(e)
            rid=self.create_rectangle(x0,y0,x1,y0+pedal_h, fill="#1d3f2a", outline="#86c591", tags=("ped",))
            self._ped_map.append((rid,i))
            self.create_rectangle(x0-3,y0,x0+3,y0+pedal_h, fill="#86c591", outline="")
            self.create_rectangle(x1-3,y0,x1+3,y0+pedal_h, fill="#86c591", outline="")

    def _hit_note(self, x, y):
        ids=self.find_overlapping(x,y,x,y)
        note_id=None; grip=False; voice=None; idx=None
        for i in ids:
            tags=self.gettags(i)
            if "grip" in tags: grip=True
            if "note" in tags or "grip" in tags:
                for t in tags:
                    if t in VOICE_ORDER: voice=t; break
                if "note" in tags: note_id=i
        if note_id is None: return None
        v,i = self._map.get(note_id, (None,None))
        if v is None: v=voice
        return {"id":note_id,"voice":v,"index":i,"grip":grip}

    def _pedal_edge_hit(self, x, y):
        if not self._sustain: return None
        main_h = self.margin_top + (self.max_pitch - self.min_pitch + 1) * self.pitch_px + 20
        y0 = main_h + 6
        if y < y0:
            return None
        b=self.x_to_beat(x)
        for i,(s,e) in enumerate(self._sustain):
            if abs(self.beat_to_x(s)-x)<=6: return (i,"L")
            if abs(self.beat_to_x(e)-x)<=6: return (i,"R")
            if s <= b <= e: return (i,None)
        return None

    def on_left_down(self, ev):
        if not self._score: return
        x,y=ev.x,ev.y; grid=self.get_grid()
        main_h=self.margin_top+(self.max_pitch-self.min_pitch+1)*self.pitch_px+20
        if y >= main_h+6:
            ped = self._pedal_edge_hit(x,y)
            if ped:
                i, edge = ped
                if edge in ("L","R"):
                    self._drag_pedal={"mode":"resize","index":i,"edge":edge}
                    return
            self._drag_pedal={"mode":"new","start": q(self.x_to_beat(x), grid)}
            return
        hit=self._hit_note(x,y)
        if hit:
            v, idx = hit["voice"], hit["index"]
            self._selected=(v,idx); self.set_active_voice(v)
            ws=self._score.parts[v].with_starts(); st, n = ws[idx]
            self._hit={"voice":v,"index":idx,"start":st,"dur":n.beats,"pitch":n.pitch,"vel":n.vel,
                       "mode":("resize" if hit["grip"] else "move"),"x0":x,"y0":y}
            self.focus_set(); return
        self._hit={"mode":"empty","x0":x,"y0":y}

    def on_left_drag(self, ev):
        if not self._score: return
        x,y=ev.x,ev.y; grid=self.get_grid()
        if self._drag_pedal is not None:
            if self._drag_pedal["mode"]=="resize":
                i=self._drag_pedal["index"]; edge=self._drag_pedal["edge"]
                s,e=self._sustain[i]; b=q(self.x_to_beat(x), grid)
                if edge=="L": s=min(b,e-0.01)
                else: e=max(b,s+0.01)
                self._sustain[i]=(s,e); self._score.sustain=self._sustain
                self.set_score(self._score, self._sustain)
            return
        if not self._hit or self._hit.get("mode") not in ("move","resize"): return
        info=self._hit; v=info["voice"]; idx=info["index"]
        part=self._score.parts[v]; ws=part.with_starts()
        st, n = ws[idx]; start=info["start"]; dur=info["dur"]; pitch=info["pitch"]; vel=info["vel"]
        prev_end=0.0 if idx==0 else ws[idx-1][0]+ws[idx-1][1].beats
        next_start = (ws[idx+1][0] if idx+1 < len(ws) else self._score.total_beats())
        if info["mode"]=="move":
            dx=q(self.x_to_beat(x)-self.x_to_beat(info["x0"]), grid)
            dy=int((info["y0"]-y)/self.pitch_px)
            new_start=q(clamp(start+dx, prev_end, next_start-dur), grid)
            new_pitch=clamp(pitch+dy, self.min_pitch, self.max_pitch)
            self._apply_edit_move(v, idx, new_start, dur, new_pitch, vel)
            self._hit.update({"x0":x,"y0":y,"start":new_start,"pitch":new_pitch})
        else:
            right=start+dur; new_right=q(max(self.x_to_beat(x), start+grid), grid)
            push = (ev.state & 0x0004) != 0
            if not push: new_right=min(new_right, next_start)
            new_dur=max(grid, new_right-start)
            self._apply_edit_resize(v, idx, start, new_dur, pitch, vel, push=push)
            self._hit.update({"x0":x,"dur":new_dur})

    def on_left_up(self, ev):
        if self._drag_pedal is not None:
            grid=self.get_grid()
            if self._drag_pedal["mode"]=="new":
                b0=self._drag_pedal["start"]; b1=q(self.x_to_beat(ev.x), grid)
                if b1!=b0:
                    s,e=(b0,b1) if b0<b1 else (b1,b0)
                    self._merge_pedal((s,e))
            self._merge_pedal()
            self._drag_pedal=None
            self.on_modified(self._score)
            return
        if self._hit and self._hit.get("mode") in ("move","resize"):
            self.on_modified(self._score)
        self._hit=None

    def on_double_left(self, ev):
        if not self._score: return
        v=self.get_active_voice(); grid=self.get_grid()
        b=q(self.x_to_beat(ev.x), grid); p=self.y_to_pitch(ev.y)
        self._insert_note(v, b, 1.0, p, 96)
        self._selected=(v, self._find_index_by_time(v, b))
        self.on_modified(self._score)

    def on_right_click(self, ev):
        ped=self._pedal_edge_hit(ev.x, ev.y)
        if ped and ped[0] is not None:
            i,_=ped
            self._sustain.pop(i); self._score.sustain=self._sustain
            self.set_score(self._score, self._sustain)
            self.on_modified(self._score); return
        hit=self._hit_note(ev.x, ev.y)
        if hit:
            v, idx = hit["voice"], hit["index"]
            self._delete_note(v, idx); self._selected=None; self.on_modified(self._score)

    def on_key_delete(self, _ev):
        if not self._selected: return
        v, idx = self._selected
        self._delete_note(v, idx); self._selected=None; self.on_modified(self._score)

    def on_shift_drag(self, ev):
        if not self._selected: return
        v, idx = self._selected; part=self._score.parts[v]; ws=part.with_starts()
        st,n=ws[idx]
        base = getattr(self, "_vel_drag_base", None)
        if base is None:
            self._vel_drag_base = (ev.y, n.vel)
        y0, v0 = self._vel_drag_base
        new_vel = int(clamp(v0 + (y0-ev.y)/2, 1, 127))
        n.vel=new_vel; self._score.parts[v].notes[idx]=n
        self.set_score(self._score, self._sustain)

    # ---- 编辑实现（与 v3.1 相同思路） ----
    def _apply_edit_move(self, v, idx, new_start, dur, pitch, vel):
        part=self._score.parts[v]; ws=part.with_starts(); total=self._score.total_beats()
        new_notes=[]; cur=0.0
        for i,(st,n) in enumerate(ws):
            if i==idx:
                if new_start>cur: new_notes.append(Note(None, new_start-cur))
                new_notes.append(Note(pitch, dur, vel)); cur=new_start+dur
            else:
                if st>=cur: new_notes.append(Note(n.pitch, n.beats, n.vel)); cur=st+n.beats
        if cur<total: new_notes.append(Note(None, total-cur))
        part.notes=new_notes; self.set_score(self._score, self._sustain)

    def _apply_edit_resize(self, v, idx, start, new_dur, pitch, vel, push=False):
        part=self._score.parts[v]; ws=part.with_starts(); total=self._score.total_beats()
        delta=new_dur - ws[idx][1].beats
        new_notes=[]; cur=0.0
        for i,(st,n) in enumerate(ws):
            if i==idx:
                if st>cur: new_notes.append(Note(None, st-cur))
                new_notes.append(Note(pitch, new_dur, vel)); cur=st+new_dur
            else:
                if push and i>idx and delta!=0:
                    st_shift = st + delta
                    if st_shift>cur: new_notes.append(Note(None, st_shift-cur))
                    new_notes.append(Note(n.pitch, n.beats, n.vel)); cur=st_shift+n.beats
                else:
                    if st>cur: new_notes.append(Note(None, st-cur))
                    new_notes.append(Note(n.pitch, n.beats, n.vel)); cur=st+n.beats
        if cur<max(total, ws[-1][0]+ws[-1][1].beats + max(0,delta)): new_notes.append(Note(None, (max(total, cur)-cur)))
        part.notes=new_notes; self.set_score(self._score, self._sustain)

    def _insert_note(self, v, beat, dur, pitch, vel):
        part=self._score.parts[v]; ws=part.with_starts(); total=self._score.total_beats()
        grid=self.get_grid(); beat=q(beat,grid); dur=max(grid,q(dur,grid))
        new=[]; cur=0.0; inserted=False
        for st,n in ws:
            if not inserted and beat<=st:
                if beat>cur: new.append(Note(None, beat-cur))
                new.append(Note(pitch, dur, vel)); cur=beat+dur; inserted=True
            if st>=cur: new.append(Note(n.pitch, n.beats, n.vel)); cur=st+n.beats
        if not inserted:
            if beat>cur: new.append(Note(None, beat-cur))
            new.append(Note(pitch, dur, vel)); cur=beat+dur
        if cur<total: new.append(Note(None, total-cur))
        part.notes=new; self.set_score(self._score, self._sustain)

    def _delete_note(self, v, idx):
        part=self._score.parts[v]; ws=part.with_starts(); total=self._score.total_beats()
        new=[]; cur=0.0
        for i,(s,n) in enumerate(ws):
            if i==idx: continue
            if s>cur: new.append(Note(None, s-cur))
            new.append(Note(n.pitch, n.beats, n.vel)); cur=s+n.beats
        if cur<total: new.append(Note(None, total-cur))
        part.notes=new; self.set_score(self._score, self._sustain)

    def _find_index_by_time(self, v, t):
        ws=self._score.parts[v].with_starts()
        for i,(st,n) in enumerate(ws):
            if st<=t<st+n.beats: return i
        return min(range(len(ws)), key=lambda i:abs(ws[i][0]-t)) if ws else 0

    def _merge_pedal(self, extra=None):
        ivs=self._sustain[:]
        if extra: ivs.append(extra)
        ivs.sort(); merged=[]
        for s,e in ivs:
            if not merged or s>merged[-1][1]+1e-6: merged.append([s,e])
            else: merged[-1][1]=max(merged[-1][1], e)
        self._sustain=[(a,b) for a,b in merged]
        self._score.sustain=self._sustain
        self.set_score(self._score, self._sustain)

    def start_cursor(self,bpm,total_beats):
        self._bpm=bpm; self.total_beats=total_beats
        self._cursor_start_ts=time.perf_counter(); self._cursor_running=True; self._tick()
    def stop_cursor(self): self._cursor_running=False
    def _tick(self):
        if not self._cursor_running: return
        el=time.perf_counter()-self._cursor_start_ts
        cur= el / beats_to_seconds(1,self._bpm)
        x=self.beat_to_x(cur)
        main_h=self.margin_top+(self.max_pitch-self.min_pitch+1)*self.pitch_px+20
        self.coords(self.cursor, x,self.margin_top,x,main_h-10)
        if self._scroll_follow:
            wv=self.winfo_width(); denom=max(1,(self._scroll_w-wv))
            self.xview_moveto(max(0,(x-self.margin_left-wv*0.4)/denom))
        if cur<=self.total_beats+0.2: self.after(30,self._tick)
        else: self.stop_cursor()

# ====================== 音频后端（与 v3.1 相同） ======================
class AudioBackendBase: name="Base"
def _safe_join(th): 
    if th and th.is_alive(): th.join(timeout=1.0)

class FluidSynthBackend(AudioBackendBase):
    name="SoundFont (FluidSynth)"
    def __init__(self): self.fs=None; self.sf_path=None; self.thread=None; self.stop_evt=threading.Event(); self.used_driver=None; self.used_preset=(0,0)
    def load_soundfont(self,p): self.sf_path=p
    def _pick_driver(self):
        drivers=[None,"wasapi","dsound","winmm","portaudio","sdl2"]; last=None
        for d in drivers:
            try:
                fs=fluidsynth.Synth(samplerate=44100); fs.start() if d is None else fs.start(driver=d)
                self.used_driver=d or "default"; return fs
            except Exception as e: last=e
        raise RuntimeError(f"FluidSynth 启动失败：{last}")
    def _select_first_preset(self,sfid):
        for bank in (0,1,128):
            for pr in range(128):
                try:
                    for ch in range(4): self.fs.program_select(ch, sfid, bank, pr)
                    self.used_preset=(bank,pr); return
                except: continue
        for ch in range(4): self.fs.program_select(ch, sfid, 0, 0); self.used_preset=(0,0)
    def prepare(self):
        if not HAVE_FS: raise RuntimeError("未安装 pyfluidsynth")
        if not self.sf_path or not os.path.exists(self.sf_path): raise RuntimeError("未选择有效 .sf2")
        self.fs=self._pick_driver(); sfid=self.fs.sfload(self.sf_path)
        if sfid<0: raise RuntimeError(f"加载 SoundFont 失败: {self.sf_path}")
        self._select_first_preset(sfid)
    def _run(self, events):
        t0=time.perf_counter(); active=set()
        for (t,on,p,ch,vel) in events:
            if self.stop_evt.is_set(): break
            now=time.perf_counter()-t0
            if t>now: time.sleep(max(0.0,t-now))
            if self.stop_evt.is_set(): break
            if p==-1 and on:
                try: self.fs.cc(ch,64,int(vel))
                except: pass
                continue
            if on: self.fs.noteon(ch,p,max(1,vel)); active.add((ch,p))
            else:
                self.fs.noteoff(ch,p); active.discard((ch,p))
        for ch,p in list(active):
            try: self.fs.noteoff(ch,p)
            except: pass
    def start(self, events):
        self.stop_evt.clear(); self.thread=threading.Thread(target=self._run,args=(events,),daemon=True); self.thread.start()
    def stop(self):
        self.stop_evt.set(); _safe_join(self.thread)
        if self.fs:
            try: self.fs.delete()
            except: pass
        self.fs=None

class SampleInstrument:
    def __init__(self): self.samples={}; self.cache={}
    @staticmethod
    def _parse_pitch_from_name(name):
        m=re.search(r"([A-Ga-g])([#b]?)(-?\d)\.wav$", name)
        if not m: return None
        nm=m.group(1).upper()+m.group(2); octv=int(m.group(3))
        return 12*(octv+1)+NOTE_MAP.get(nm,0)
    def load_folder(self, folder):
        ok=0
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100,-16,2,512); pygame.mixer.init()
        for f in os.listdir(folder):
            if f.lower().endswith(".wav"):
                p=self._parse_pitch_from_name(f); 
                if p is None: continue
                path=os.path.join(folder,f); snd=pygame.mixer.Sound(path)
                arr=pygame.sndarray.array(snd).astype(np.int16)
                if arr.ndim==1: arr=arr[:,None]
                self.samples[p]=(arr, pygame.mixer.get_init()[0], arr.shape[1]); ok+=1
        return ok
    def _nearest(self, target):
        if not self.samples: return None
        return min(self.samples.keys(), key=lambda sp:abs(sp-target))
    def get_sound_for(self, pitch):
        if pitch in self.cache: return self.cache[pitch]
        if not self.samples: return None
        src=self._nearest(pitch); arr, sr, ch = self.samples[src]
        ratio=2**((pitch-src)/12.0); n_out=int(arr.shape[0]/ratio)
        x=np.linspace(0, arr.shape[0]-1, n_out); xi=np.floor(x).astype(int); xf=x-xi; xi1=np.clip(xi+1,0,arr.shape[0]-1)
        out=(arr[xi]*(1-xf)[:,None]+arr[xi1]*xf[:,None]).astype(np.int16)
        snd=pygame.sndarray.make_sound(out); self.cache[pitch]=snd; return snd

class SamplerBackend(AudioBackendBase):
    name="Sampler (WAV)"
    def __init__(self): self.inst=SampleInstrument(); self.thread=None; self.stop_evt=threading.Event()
    def load_folder(self, folder): return self.inst.load_folder(folder)
    def prepare(self):
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100,-16,2,512); pygame.mixer.init()
    def _run(self, events):
        t0=time.perf_counter(); playing={}; sustain_on=[False]*16; sustained={}
        for (t,on,p,ch,vel) in events:
            if self.stop_evt.is_set(): break
            now=time.perf_counter()-t0
            if t>now: time.sleep(max(0.0,t-now))
            if self.stop_evt.is_set(): break
            key=(ch,p)
            if p==-1 and on:
                sustain_on[ch]=(vel>=64)
                if not sustain_on[ch]:
                    for k,chan in list(sustained.items()):
                        try: chan.fadeout(40)
                        except: pass
                        sustained.pop(k,None)
                continue
            if on:
                snd=self.inst.get_sound_for(p)
                if snd is None: continue
                chan=pygame.mixer.find_channel(True)
                if chan:
                    chan.set_volume(max(0.05, vel/127.0))
                    chan.play(snd); playing[key]=chan
            else:
                chan=playing.pop(key,None)
                if chan:
                    if sustain_on[ch]: sustained[key]=chan
                    else: chan.fadeout(40)
        pygame.mixer.stop()
    def start(self, events):
        self.stop_evt.clear(); self.thread=threading.Thread(target=self._run,args=(events,),daemon=True); self.thread.start()
    def stop(self):
        self.stop_evt.set(); _safe_join(self.thread)
        try: pygame.mixer.stop()
        except: pass

class SystemMIDIVanilla(AudioBackendBase):
    name="System MIDI"
    def __init__(self): self.thread=None; self.stop_evt=threading.Event(); self.dev=None; self.dev_id=None
    def _pick(self):
        d=pygame.midi.get_default_output_id()
        if d!=-1: return d
        n=pygame.midi.get_count()
        for i in range(n):
            inf=pygame.midi.get_device_info(i); 
            if inf and inf[3]: return i
        return -1
    def prepare(self):
        pygame.midi.init(); dev_id=self._pick()
        if dev_id==-1:
            pygame.midi.quit(); raise RuntimeError("没有可用的 System MIDI 输出设备")
        self.dev_id=dev_id; self.dev=pygame.midi.Output(dev_id)
        for ch in range(4):
            self.dev.write_short(0xB0+ch,121,0); self.dev.write_short(0xB0+ch,7,127); self.dev.set_instrument(0, ch)
    def _run(self, events):
        t0=time.perf_counter(); active=set()
        try:
            for (ts,on,p,ch,vel) in events:
                if self.stop_evt.is_set(): break
                now=time.perf_counter()-t0
                if ts>now: time.sleep(max(0.0, ts-now))
                if self.stop_evt.is_set(): break
                if p==-1 and on: self.dev.write_short(0xB0+ch,64,int(vel)); continue
                if on: self.dev.note_on(int(p), max(1,int(vel)), int(ch)); active.add((ch,p))
                else:
                    self.dev.note_off(int(p), 0, int(ch)); active.discard((ch,p))
        finally:
            for ch,p in list(active):
                try: self.dev.note_off(int(p),0,int(ch))
                except: pass
    def start(self, events):
        self.stop_evt.clear(); self.thread=threading.Thread(target=self._run,args=(events,),daemon=True); self.thread.start()
    def stop(self):
        self.stop_evt.set(); _safe_join(self.thread)
        if self.dev:
            try:
                for ch in range(4): self.dev.write_short(0xB0+ch,123,0); self.dev.write_short(0xB0+ch,64,0)
            except: pass
            try: self.dev.close()
            except: pass
        self.dev=None; self.dev_id=None
        try: pygame.midi.quit()
        except: pass

# ====================== GUI 应用 ======================
class App:
    def __init__(self, root):
        self.root=root; root.title(APP_TITLE); root.geometry("1280x880"); self._style()
        self.backend_name=tk.StringVar(value="SoundFont (FluidSynth)" if HAVE_FS else "Sampler (WAV)")
        self.bpm=tk.IntVar(value=DEFAULT_BPM); self.transpose=tk.IntVar(value=0)
        self.sf2_path=tk.StringVar(value=""); self.sample_dir=tk.StringVar(value="")
        self.grid_res=tk.DoubleVar(value=GRID_DEFAULT); self.active_voice=tk.StringVar(value="S")
        self.backend=None; self._current_score=None
        self._build_ui(); self._load_demo()

    def _style(self):
        s=ttk.Style(); th="clam" if "clam" in s.theme_names() else s.theme_use()
        s.theme_use(th); s.configure("TButton", padding=6)
        s.configure("Header.TLabel", font=("Segoe UI", 15,"bold"))
        s.configure("Caption.TLabel", foreground="#777")

    def _build_ui(self):
        top=ttk.Frame(self.root); top.pack(fill="x", padx=12,pady=8)
        ttk.Label(top, text="SATB 四声部编曲器", style="Header.TLabel").pack(side="left")
        ttk.Label(top, text="  卷帘编辑 · 音符属性(dyn/vel/ped) · CC64 · 规则检查", style="Caption.TLabel").pack(side="left")

        bar=ttk.Frame(self.root); bar.pack(fill="x", padx=12)
        ttk.Label(bar, text="BPM").pack(side="left")
        ttk.Spinbox(bar, from_=30,to=240, width=5, textvariable=self.bpm).pack(side="left", padx=(4,12))
        ttk.Label(bar, text="移调").pack(side="left")
        ttk.Spinbox(bar, from_=-24,to=24, width=5, textvariable=self.transpose).pack(side="left", padx=(4,12))

        ttk.Label(bar, text="网格").pack(side="left", padx=(8,2))
        cbgrid=ttk.Combobox(bar, state="readonly", width=8, values=["1/4","1/8","1/16"]); cbgrid.current(2); cbgrid.pack(side="left")
        cbgrid.bind("<<ComboboxSelected>>", lambda e:self.grid_res.set({"1/4":1.0,"1/8":0.5,"1/16":0.25}[cbgrid.get()]))

        ttk.Label(bar, text="编辑声部").pack(side="left", padx=(12,2))
        ttk.Combobox(bar, state="readonly", width=5, textvariable=self.active_voice, values=VOICE_ORDER).pack(side="left")

        ttk.Label(bar, text="音频后端").pack(side="left", padx=(12,2))
        ttk.Combobox(bar, state="readonly", width=22, textvariable=self.backend_name,
                     values=["SoundFont (FluidSynth)","Sampler (WAV)","System MIDI"]).pack(side="left")

        ttk.Button(bar, text="选择 SoundFont", command=self.pick_sf2).pack(side="left", padx=6)
        ttk.Button(bar, text="选择采样目录", command=self.pick_sample_dir).pack(side="left", padx=6)
        ttk.Button(bar, text="刷新/检查", command=self.refresh_and_check).pack(side="left", padx=(16,6))
        ttk.Button(bar, text="▶ 播放", command=self.play).pack(side="left", padx=(16,6))
        ttk.Button(bar, text="■ 停止", command=self.stop).pack(side="left", padx=6)
        ttk.Button(bar, text="💾 导出 MIDI", command=self.export_midi).pack(side="left", padx=6)

        self.status=tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x", padx=12, pady=(2,6))

        paned=ttk.Panedwindow(self.root, orient="horizontal"); paned.pack(fill="both", expand=True, padx=12, pady=6)
        left=ttk.Frame(paned); right=ttk.Frame(paned); paned.add(left, weight=1); paned.add(right, weight=2)

        self.texts={}
        for v in VOICE_ORDER:
            row=ttk.Frame(left); row.pack(fill="x", pady=5)
            ttk.Label(row, text=f"{v}").pack(side="left")
            t=tk.Text(row, height=5, wrap="word"); t.pack(side="left", fill="x", expand=True)
            self.texts[v]=t

        gen=ttk.Labelframe(left, text="从和弦自动生成（基础连贯）"); gen.pack(fill="x", pady=8)
        self.chords=tk.StringVar(value="C F G7 C | Am Dm G7 C"); self.durs=tk.StringVar(value="q q q q q q q q")
        ttk.Label(gen, text="和弦：").grid(row=0,column=0,sticky="w"); ttk.Entry(gen,textvariable=self.chords).grid(row=0,column=1,sticky="we", padx=6)
        ttk.Label(gen, text="时值：").grid(row=1,column=0,sticky="w"); ttk.Entry(gen,textvariable=self.durs).grid(row=1,column=1,sticky="we", padx=6)
        btns=ttk.Frame(gen); btns.grid(row=0,column=2,rowspan=2, padx=6)
        ttk.Button(btns, text="覆盖填入", command=self.gen_chords_overwrite).pack(pady=2)
        ttk.Button(btns, text="在尾部追加", command=lambda:self.gen_chords_overwrite(append=True)).pack(pady=2)
        gen.columnconfigure(1, weight=1)

        roll_box=ttk.Labelframe(right, text="钢琴卷帘（拖动=移动；抓右端=改时值；Ctrl+拉伸=推挤；双击空白=新增；右键=删除；Shift拖=力度；下方=延音）")
        roll_box.pack(fill="both", expand=True)
        self.roll=PianoRollCanvas(roll_box, on_modified=self.on_score_modified,
                                  get_active_voice=lambda:self.active_voice.get(),
                                  set_active_voice=self.active_voice.set,
                                  get_grid=lambda:self.grid_res.get())
        self.roll.pack(fill="both", expand=True)

        issues_box=ttk.Labelframe(right, text="规则问题"); issues_box.pack(fill="both", expand=False, pady=6)
        cols=("beat","type","detail"); self.tree=ttk.Treeview(issues_box, columns=cols, show="headings", height=8)
        for c in cols: self.tree.heading(c,text=c)
        self.tree.column("beat", width=80, anchor="center"); self.tree.column("type", width=100, anchor="center"); self.tree.column("detail", width=640, anchor="w")
        self.tree.pack(fill="x", expand=False); self.tree.bind("<<TreeviewSelect>>", self.on_select_issue)

    def _load_demo(self):
        demo={
            "S":"C5q{dyn=mp,ped=on} D5q E5q F5q{vel=100} | G5h A5q G5q{ped=off} | A5q{dyn=mf,ped=on} G5q F5q E5q | D5h C5h{dyn=p,ped=off}",
            "A":"A4h.{dyn=mp} Rq | B4q C5q B4q A4q | G4h{vel=80} F4h | E4q F4q G4q A4q",
            "T":"E4h E4h | F4q G4q F4q E4q | D4h. C4q | B3q{dyn=f} C4q D4q E4q",
            "B":"C3w | C3h G2h{vel=90} | F2w | C3w"
        }
        for v in VOICE_ORDER:
            self.texts[v].delete("1.0","end"); self.texts[v].insert("1.0", demo[v])
        self.refresh_and_check()

    # ----- 文本 <-> Score（含 attrs 与 pedal 解析/回写） -----
    def _score_from_texts(self, bpm=None):
        sc=Score(bpm if bpm else self.bpm.get(), DEFAULT_METER)
        toggle_times=[]  # (time, val)  val: 127/0
        for v in VOICE_ORDER:
            text=self.texts[v].get("1.0","end").replace("|"," ")
            toks=[t for t in text.split() if t.strip()]
            t=0.0; seq=[]
            for tok in toks:
                p,b,attr = parse_token(tok)
                # 力度
                vel=96
                if "vel" in attr:
                    try: vel=int(attr["vel"]); vel=int(clamp(vel,1,127))
                    except: vel=96
                elif "dyn" in attr and attr["dyn"] in DYN_TO_VEL:
                    vel=DYN_TO_VEL[attr["dyn"]]
                seq.append(Note(p, b, vel))
                # 踏板开关（允许写在任意声部的该拍）
                if "ped" in attr:
                    val = 127 if attr["ped"] in ("on","down","1","true") else 0
                    toggle_times.append((t, val))
                t += b
            sc.parts[v].extend(seq)
        sc.ensure_aligned()
        # toggle -> 区段
        sc.sustain = self._toggles_to_intervals(toggle_times, sc.total_beats())
        return sc

    @staticmethod
    def _toggles_to_intervals(toggles, total):
        # 合并不同声部的开关；顺序化；on->off 配对
        if not toggles: return []
        # 去重：同一时刻重复 on/off 只算一次
        uniq={}
        for tt, val in toggles:
            key=(round(tt,6), val)  # 简单量化
            uniq[key]=tt
        items=sorted([(t,val) for (rt,val),t in uniq.items()], key=lambda x:x[0])
        ivs=[]; ped_on=None
        for t,val in items:
            if val>=64:
                ped_on = t
            else:
                if ped_on is not None and t>ped_on:
                    ivs.append((ped_on, t)); ped_on=None
        if ped_on is not None and total>ped_on:
            ivs.append((ped_on, total))
        return ivs

    def _texts_from_score(self, sc:Score):
        # 先计算 S 声部上需要标记的 ped on/off 时刻（量化到网格）
        grid = float(self.grid_res.get())
        ped_on  = {round(q(s, grid),6) for (s,_) in sc.sustain}
        ped_off = {round(q(e, grid),6) for (_,e) in sc.sustain}

        for v in VOICE_ORDER:
            ws=sc.parts[v].with_starts()
            toks=[]
            for st,n in ws:
                # 基本 token
                if n.pitch is None:
                    base=f"R{dur_to_token(n.beats)}"
                else:
                    base=f"{format_pitch(n.pitch)}{dur_to_token(n.beats)}"

                # 力度 -> dyn 或 vel
                attrs=[]
                if n.pitch is not None:
                    kind,val = vel_to_dyn_or_num(int(n.vel))
                    if kind=="dyn": attrs.append(f"dyn={val}")
                    else: attrs.append(f"vel={val}")

                # 只有 S 声部在音符起点写 ped 标记（避免四处重复）
                if v=="S":
                    rst = round(q(st, grid),6)
                    if rst in ped_on:  attrs.append("ped=on")
                    if rst in ped_off: attrs.append("ped=off")

                if attrs:
                    toks.append(f"{base}{{{','.join(attrs)}}}")
                else:
                    toks.append(base)

            # 简单每 4 拍插一条竖线
            self.texts[v].delete("1.0","end")
            total=0.0; out=[]
            for t in toks:
                out.append(t)
                m=re.search(r"(\d+(?:\.\d+)?)\}?\s*$", t)  # 取末尾数值时值
                d=float(m.group(1)) if m else DUR_MAP.get(t[-1],1.0)
                total+=d
                if abs(total%4.0)<1e-6: out.append("|")
            self.texts[v].insert("1.0"," ".join(out).replace("| |","|"))

    # ----- 后端 -----
    def _build_backend(self):
        name=self.backend_name.get()
        if name.startswith("SoundFont"): be=FluidSynthBackend(); be.load_soundfont(self.sf2_path.get()); return be
        if name.startswith("Sampler"):
            be=SamplerBackend()
            if self.sample_dir.get():
                n=be.load_folder(self.sample_dir.get())
                if n==0: messagebox.showwarning("采样器","目录中未找到如 piano_C4.wav；将改为 System MIDI 兜底。")
            return be
        return SystemMIDIVanilla()

    # ----- 播放/停止/导出/检查 -----
    def play(self):
        self.stop()
        sc=self._current_score or self._score_from_texts(self.bpm.get())
        events=score_to_events(sc, transpose=self.transpose.get())
        be=self._build_backend()
        try: be.prepare()
        except Exception as e:
            if not isinstance(be, SystemMIDIVanilla):
                try:
                    fb=SystemMIDIVanilla(); fb.prepare(); be=fb; self.backend_name.set("System MIDI")
                    self.status.set(f"后端失败，已降级为 System MIDI：{e}")
                except Exception as e2:
                    messagebox.showerror("无法播放", f"{e}\n且兜底后端也不可用：{e2}"); return
            else:
                messagebox.showerror("无法播放", str(e)); return
        self.roll.set_score(sc, sustain=sc.sustain); self.roll.start_cursor(sc.bpm, sc.total_beats())
        self.backend=be; self.backend.start(events)
        self.status.set(f"播放中（{sc.bpm} BPM, 移调 {self.transpose.get()}） | 后端: {be.name}")

    def stop(self):
        if self.backend:
            try: self.backend.stop()
            except: pass
        self.backend=None; 
        if hasattr(self,"roll"): self.roll.stop_cursor()
        self.status.set("已停止。")

    def export_midi(self):
        sc=self._current_score or self._score_from_texts(self.bpm.get())
        fn=filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI File","*.mid")], initialfile="satb.mid")
        if not fn: return
        mf=MIDIFile(5)   # 0..4：0 用作全局/踏板，1..4 对应 S/A/T/B
        # 轨道 0：Tempo + 踏板 CC
        mf.addTempo(0, 0, sc.bpm)
        for (s,e) in sc.sustain:
            mf.addControllerEvent(0, 0, s, 64, 127)
            mf.addControllerEvent(0, 0, e, 64,   0)
        # 四声部
        for i,v in enumerate(VOICE_ORDER, start=1):
            ch=VOICE_TO_CH[v]
            mf.addTempo(i, 0, sc.bpm); mf.addProgramChange(i, ch, 0, 0)
            t=0.0
            for n in sc.parts[v].notes:
                if n.pitch is not None:
                    p=n.pitch+self.transpose.get()
                    mf.addNote(i, ch, p, t, n.beats, int(n.vel))
                t+=n.beats
        with open(fn,"wb") as f: mf.writeFile(f)
        self.status.set(f"已导出 MIDI：{os.path.basename(fn)}")

    def refresh_and_check(self):
        sc=self._current_score or self._score_from_texts(self.bpm.get())
        self.roll.set_score(sc, sustain=sc.sustain)
        issues=RuleChecker(sc).check()
        for i in self.tree.get_children(): self.tree.delete(i)
        for it in issues:
            self.tree.insert("", "end", values=(f"{it['beat']:.2f}", it["type"], it["detail"]))
        if issues: self.status.set(f"发现 {len(issues)} 个问题（点击列表可定位）")
        else: self.status.set("规则检查通过 ✓")

    def on_select_issue(self, _e):
        sel=self.tree.selection()
        if not sel: return
        b=float(self.tree.item(sel[0])["values"][0]); self.roll.scroll_to_beat(b)

    def on_score_modified(self, sc_from_roll: Score):
        # 卷帘修改后：直接把力度/延音写回 SATB 文本（含属性）
        self._current_score=sc_from_roll
        self._texts_from_score(sc_from_roll)   # <- 现在会写 dyn/vel 与 ped 标注
        # 同时做规则检查
        self.refresh_and_check()

    def gen_chords_overwrite(self, append=False):
        prog=self.chords.get().replace("|"," "); chords=[c for c in prog.split() if c.strip()]
        durs=[d for d in self.durs.get().split() if d.strip()]
        try: parts=Arranger.chord_prog_to_satb(chords, durs)
        except Exception as e: messagebox.showerror("生成失败", str(e)); return
        sc=self._current_score or self._score_from_texts(self.bpm.get())
        for v in VOICE_ORDER:
            seq=parts[v]
            if append: sc.parts[v].notes += seq
            else: sc.parts[v].notes  = seq
        sc.ensure_aligned(); self._current_score=sc
        self._texts_from_score(sc); self.refresh_and_check()

    def pick_sf2(self):
        p=filedialog.askopenfilename(filetypes=[("SoundFont","*.sf2"),("All files","*.*")])
        if p: self.sf2_path.set(p); self.status.set(f"SoundFont: {os.path.basename(p)}")
    def pick_sample_dir(self):
        d=filedialog.askdirectory()
        if d: self.sample_dir.set(d); self.status.set(f"采样目录: {d}")

# ====================== 入口 ======================
def main():
    pygame.init()
    root=tk.Tk()
    App(root)
    root.mainloop()

if __name__=="__main__":
    main()
