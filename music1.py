# SATB Arranger - Version 1
# OOP + GUI + SoundFont / Sampler / System MIDI
# 功能：四声部编辑、播放、导出MIDI、和弦->四声部（简版）
# 已修复：Windows 下 FluidSynth 驱动/预设 & System MIDI 无声、首播需点停止

import os, sys, re, time, threading, tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pygame
import pygame.midi
import pygame.sndarray
from midiutil import MIDIFile

# ---------------- Windows: 指定 FluidSynth DLL 目录（自动探测） ----------------
if sys.platform == "win32":
    # 先试环境变量 FLUIDSYNTH_BIN，再试常用路径
    for _cand in [os.getenv("FLUIDSYNTH_BIN"), r"C:\tools\fluidsynth\bin"]:
        if _cand and os.path.isdir(_cand):
            try:
                os.add_dll_directory(_cand)
                break
            except Exception:
                pass

# ---------------- 可选导入 FluidSynth ----------------
HAVE_FS = False
try:
    import fluidsynth
    HAVE_FS = True
except Exception:
    HAVE_FS = False

APP_TITLE = "SATB Arranger — v1"

# ====================== 音乐基础 & 工具 ======================
NOTE_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11
}
DUR_MAP = { "w":4.0, "h":2.0, "q":1.0, "e":0.5, "s":0.25, "t":0.125 }
VOICE_ORDER = ["S","A","T","B"]
VOICE_TO_CH = {v:i for i,v in enumerate(VOICE_ORDER)}
RANGES = { "S":(60,84), "A":(55,77), "T":(48,69), "B":(40,60) }  # 建议音域
def clamp(v, lo, hi): return max(lo, min(hi, v))
def beats_to_seconds(beats, bpm): return (60.0/bpm)*beats

def note_to_midi(token: str):
    if token.upper().startswith("R"): return None
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)", token.strip())
    if not m: raise ValueError(f"音名错误: {token}")
    name = m.group(1).upper()+m.group(2); octv = int(m.group(3))
    if name not in NOTE_MAP: raise ValueError(f"不认识的音名: {name}")
    return 12*(octv+1)+NOTE_MAP[name]

def parse_token(tok: str):
    # e.g. C4q, Bb3h., Rq, G#5e..
    tok = tok.strip()
    if not tok: return None
    if tok[0].upper()=="R":
        m = re.fullmatch(r"R([whqest])(\.*)?", tok, re.IGNORECASE)
        if not m: raise ValueError(f"休止符错误: {tok}")
        base = DUR_MAP[m.group(1).lower()]
        dots = (m.group(2) or "")
        return (None, base*(1.5**len(dots)))
    m = re.fullmatch(r"([A-Ga-g][#b]?[-]?\d)([whqest])(\.*)?", tok)
    if not m: raise ValueError(f"音符格式错误: {tok}")
    pitch = note_to_midi(m.group(1))
    base = DUR_MAP[m.group(2).lower()]
    dots = (m.group(3) or "")
    return (pitch, base*(1.5**len(dots)))

def format_pitch(p):
    pc = p%12; octv = p//12-1
    for k,v in NOTE_MAP.items():
        if v==pc and len(k)<=2:
            return f"{k}{octv}"
    return f"C{octv}"

# ====================== 数据模型（OOP） ======================
class Note:
    __slots__ = ("pitch","beats","vel")
    def __init__(self, pitch, beats, vel=96):
        self.pitch = pitch   # None 表示休止
        self.beats = beats
        self.vel = vel

class Part:
    def __init__(self, name):
        self.name = name
        self.notes = []  # List[Note]
    def add(self, note: Note): self.notes.append(note)
    def total_beats(self): return sum(n.beats for n in self.notes)
    def extend(self, seq): self.notes.extend(seq)

class Score:
    def __init__(self, bpm=90):
        self.bpm = bpm
        self.parts = {v: Part(v) for v in VOICE_ORDER}
    def ensure_aligned(self):
        mx = max(self.parts[v].total_beats() for v in VOICE_ORDER)
        for v in VOICE_ORDER:
            t = self.parts[v].total_beats()
            if t<mx:
                self.parts[v].add(Note(None, mx-t))

# ====================== 配器（和弦→四声部） ======================
CHORD_QUAL = {
    "":  [0,4,7], "m":[0,3,7], "dim":[0,3,6], "aug":[0,4,8],
    "6":[0,4,7,9], "7":[0,4,7,10], "maj7":[0,4,7,11],
    "m7":[0,3,7,10], "m6":[0,3,7,9], "m7b5":[0,3,6,10]
}
def parse_chord(ch):
    m = re.fullmatch(r"([A-Ga-g])([#b]?)([a-zA-Z0-9#b]*)", ch.strip())
    if not m: raise ValueError(f"和弦错误: {ch}")
    root = m.group(1).upper()+m.group(2)
    qual = (m.group(3) or "").lower()
    rep = {"maj7":"maj7","maj":"maj","min":"m","min7":"m7","m7b5":"m7b5","dim7":"dim7"}
    if qual in rep: qual = rep[qual]
    if qual=="dim7": pcs = [0,3,6,9]
    else:
        if qual not in CHORD_QUAL:
            q2 = qual[:3]
            qual = q2 if q2 in CHORD_QUAL else (qual[:1] if qual[:1] in CHORD_QUAL else "")
        pcs = CHORD_QUAL[qual][:]
    root_pc = NOTE_MAP[root]
    return root_pc, [(p+root_pc)%12 for p in pcs]

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
    cands.sort(key=lambda x: abs(x-prev))
    return cands[0]

class Arranger:
    @staticmethod
    def text_to_part(text: str) -> list:
        text=text.replace("|"," ")
        toks=[t for t in text.split() if t.strip()]
        seq=[]
        for t in toks:
            p,b=parse_token(t)
            seq.append(Note(p,b))
        return seq

    @staticmethod
    def chord_prog_to_satb(chords, durs, last=None):
        beats=[]
        for d in durs:
            d=d.strip().lower()
            if d in DUR_MAP: beats.append(DUR_MAP[d])
            else: beats.append(float(d))
        if len(beats)!=len(chords):
            raise ValueError("和弦与时值数量不一致")
        prev = last or {"S":None,"A":None,"T":None,"B":None}
        parts = {v:[] for v in VOICE_ORDER}
        for ch,b in zip(chords, beats):
            _, pcs = parse_chord(ch)
            if len(pcs)==3: pcs=pcs+[pcs[0]]
            b_pc, s_pc, a_pc, t_pc = pcs[0], pcs[1%len(pcs)], pcs[2%len(pcs)], pcs[3%len(pcs)]
            bp=nearest_pitch(prev["B"], b_pc, *RANGES["B"])
            tp=nearest_pitch(prev["T"], t_pc, *RANGES["T"])
            ap=nearest_pitch(prev["A"], a_pc, *RANGES["A"])
            sp=nearest_pitch(prev["S"], s_pc, *RANGES["S"])
            parts["B"].append(Note(bp,b)); parts["T"].append(Note(tp,b))
            parts["A"].append(Note(ap,b)); parts["S"].append(Note(sp,b))
            prev.update({"B":bp,"T":tp,"A":ap,"S":sp})
        return parts

# ====================== 事件渲染 ======================
def score_to_events(score: Score, transpose=0, vel=96):
    ev=[]
    for v in VOICE_ORDER:
        ch=VOICE_TO_CH[v]; t=0.0
        for n in score.parts[v].notes:
            dur=n.beats; p=n.pitch
            sec = beats_to_seconds(dur, score.bpm)
            if p is not None:
                p2=p+transpose
                ev.append((t, True,  p2, ch, vel))
                ev.append((t+sec*0.98, False, p2, ch, 0))
            t+=sec
    ev.sort(key=lambda e:(e[0], 0 if not e[1] else 1)) # 先关后开
    return ev

# ====================== 音频后端 ======================
class AudioBackendBase:
    name="Base"
    def prepare(self): pass
    def start(self, events): pass
    def stop(self): pass

# ---- 1) FluidSynth 后端（SoundFont） ----
class FluidSynthBackend(AudioBackendBase):
    name = "SoundFont (FluidSynth)"
    def __init__(self):
        self.fs = None
        self.sf_path = None
        self.thread = None
        self.stop_evt = threading.Event()
        self.used_driver = None
        self.used_preset = (0, 0)
    def load_soundfont(self, path):
        self.sf_path = path
    def _pick_driver(self):
        drivers = [None, "wasapi", "dsound", "winmm", "portaudio", "sdl2"]
        last_err = None
        for d in drivers:
            try:
                fs = fluidsynth.Synth(samplerate=44100)
                if d is None: fs.start()
                else: fs.start(driver=d)
                self.used_driver = d or "default"
                return fs
            except Exception as e:
                last_err = e
        raise RuntimeError(f"FluidSynth 启动失败（驱动不可用）：{last_err}")
    def _select_first_preset(self, sfid):
        for bank in (0, 1, 128):
            for preset in range(128):
                try:
                    self.fs.program_select(0, sfid, bank, preset)
                    for ch in range(4):
                        self.fs.program_select(ch, sfid, bank, preset)
                    self.used_preset = (bank, preset)
                    return
                except Exception:
                    continue
        for ch in range(4):
            self.fs.program_select(ch, sfid, 0, 0)
        self.used_preset = (0, 0)
    def prepare(self):
        if not HAVE_FS:
            raise RuntimeError("未安装 pyfluidsynth：`pip install pyfluidsynth`")
        if not self.sf_path or not os.path.exists(self.sf_path):
            raise RuntimeError("未选择有效的 SoundFont (.sf2) 文件")
        self.fs = self._pick_driver()
        sfid = self.fs.sfload(self.sf_path)
        if sfid < 0:
            raise RuntimeError(f"加载 SoundFont 失败: {self.sf_path}")
        self._select_first_preset(sfid)
    def _run(self, events):
        t0 = time.perf_counter()
        active=set()
        for (t, on, pitch, ch, vel) in events:
            if self.stop_evt.is_set(): break
            now = time.perf_counter() - t0
            if t > now: time.sleep(max(0.0, t - now))
            if self.stop_evt.is_set(): break
            if on:
                self.fs.noteon(ch, pitch, max(1, vel)); active.add((ch,pitch))
            else:
                self.fs.noteoff(ch, pitch); active.discard((ch,pitch))
        for ch,p in list(active):
            try: self.fs.noteoff(ch,p)
            except: pass
    def start(self, events):
        self.stop_evt.clear()
        self.thread = threading.Thread(target=self._run, args=(events,), daemon=True)
        self.thread.start()
    def stop(self):
        self.stop_evt.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.fs:
            try: self.fs.delete()
            except: pass
        self.fs = None

# ---- 2) 采样器后端（WAV 重采样播放） ----
class SampleInstrument:
    def __init__(self):
        self.samples = {}   # pitch -> (arr(int16), sr, channels)
        self.cache   = {}   # out_pitch -> pygame Sound
    @staticmethod
    def _parse_pitch_from_name(name):
        m=re.search(r"([A-Ga-g])([#b]?)(-?\d)\.wav$", name)
        if not m: return None
        nm=m.group(1).upper()+m.group(2); octv=int(m.group(3))
        if nm not in NOTE_MAP: return None
        return 12*(octv+1)+NOTE_MAP[nm]
    def load_folder(self, folder):
        ok=0
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100, -16, 2, 512); pygame.mixer.init()
        for f in os.listdir(folder):
            if f.lower().endswith(".wav"):
                p=self._parse_pitch_from_name(f)
                if p is None: continue
                path=os.path.join(folder,f)
                snd=pygame.mixer.Sound(path)
                arr=pygame.sndarray.array(snd).astype(np.int16)
                if arr.ndim==1: arr=arr[:,None]
                self.samples[p]=(arr, pygame.mixer.get_init()[0], arr.shape[1])
                ok+=1
        return ok
    def _nearest_sample(self, target_pitch):
        if not self.samples: return None
        best=min(self.samples.keys(), key=lambda sp:abs(sp-target_pitch))
        return best
    def get_sound_for(self, pitch):
        if pitch in self.cache: return self.cache[pitch]
        if not self.samples: return None
        srcp=self._nearest_sample(pitch)
        arr, sr, ch = self.samples[srcp]  # int16
        ratio=2**((pitch-srcp)/12.0)
        n_out=int(arr.shape[0]/ratio)
        x=np.linspace(0, arr.shape[0]-1, n_out)
        xi=np.floor(x).astype(int); xf=x-xi
        xi1=np.clip(xi+1,0,arr.shape[0]-1)
        # 线性重采样到 int16
        out=(arr[xi]*(1-xf)[:,None] + arr[xi1]*xf[:,None]).astype(np.int16)
        snd=pygame.sndarray.make_sound(out)
        self.cache[pitch]=snd
        return snd

class SamplerBackend(AudioBackendBase):
    name="Sampler (WAV)"
    def __init__(self):
        self.inst = SampleInstrument()
        self.thread=None; self.stop_evt=threading.Event()
    def load_folder(self, folder):
        return self.inst.load_folder(folder)
    def prepare(self):
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100, -16, 2, 512); pygame.mixer.init()
    def _run(self, events):
        t0=time.perf_counter()
        playing={}
        for (t,on,pitch,ch,vel) in events:
            if self.stop_evt.is_set(): break
            now=time.perf_counter()-t0
            if t>now: time.sleep(max(0.0,t-now))
            if self.stop_evt.is_set(): break
            key=(ch,pitch)
            if on:
                snd=self.inst.get_sound_for(pitch)
                if snd is None: continue
                chan=pygame.mixer.find_channel(True)
                if chan:
                    chan.play(snd)
                    playing[key]=chan
            else:
                chan=playing.pop(key, None)
                if chan: chan.fadeout(40)
        pygame.mixer.stop()
    def start(self, events):
        self.stop_evt.clear()
        self.thread=threading.Thread(target=self._run,args=(events,),daemon=True)
        self.thread.start()
    def stop(self):
        self.stop_evt.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try: pygame.mixer.stop()
        except: pass

# ---- 3) System MIDI 兜底 ----
class SystemMIDIVanilla(AudioBackendBase):
    name = "System MIDI"
    def __init__(self):
        self.thread = None
        self.stop_evt = threading.Event()
        self.dev = None
        self.dev_id = None
    def _pick_output_device(self):
        default_id = pygame.midi.get_default_output_id()
        if default_id != -1: return default_id
        n = pygame.midi.get_count()
        for i in range(n):
            interf, name, is_input, is_output, opened = pygame.midi.get_device_info(i)
            if is_output: return i
        return -1
    def prepare(self):
        pygame.midi.init()
        dev_id = self._pick_output_device()
        if dev_id == -1:
            pygame.midi.quit()
            raise RuntimeError("没有可用的 System MIDI 输出设备")
        self.dev_id = dev_id
        self.dev = pygame.midi.Output(dev_id)
        # Reset / Volume / Program
        for ch in range(4):
            self.dev.write_short(0xB0 + ch, 121, 0)  # Reset All Controllers
            self.dev.write_short(0xB0 + ch, 7, 127)  # Volume -> 127
            self.dev.set_instrument(0, ch)           # Program 0 (钢琴)
    def _run(self, events):
        t0 = time.perf_counter()
        active = set()
        try:
            for (ts, on, pitch, ch, vel) in events:
                if self.stop_evt.is_set(): break
                now = time.perf_counter() - t0
                if ts > now: time.sleep(max(0.0, ts - now))
                if self.stop_evt.is_set(): break
                if on and pitch is not None:
                    self.dev.note_on(int(pitch), max(1,int(vel)), int(ch)); active.add((ch,pitch))
                elif (not on) and pitch is not None:
                    self.dev.note_off(int(pitch), 0, int(ch)); active.discard((ch,pitch))
        finally:
            for ch,p in list(active):
                try: self.dev.note_off(int(p), 0, int(ch))
                except: pass
    def start(self, events):
        self.stop_evt.clear()
        self.thread = threading.Thread(target=self._run, args=(events,), daemon=True)
        self.thread.start()
    def stop(self):
        self.stop_evt.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.dev:
            try:
                for ch in range(4):
                    self.dev.write_short(0xB0 + ch, 123, 0)  # All Notes Off
            except: pass
            try: self.dev.close()
            except: pass
        self.dev = None; self.dev_id = None
        try: pygame.midi.quit()
        except: pass

# ====================== GUI 应用 ======================
class App:
    def __init__(self, root):
        self.root=root
        root.title(APP_TITLE); root.geometry("1000x760")
        self._style()

        self.backend_name = tk.StringVar(value="SoundFont (FluidSynth)" if HAVE_FS else "Sampler (WAV)")
        self.bpm = tk.IntVar(value=90)
        self.transpose = tk.IntVar(value=0)
        self.sf2_path = tk.StringVar(value="")
        self.sample_dir = tk.StringVar(value="")
        self.backend = None
        self.playing=False

        self._build_ui()
        self._load_demo()

    def _style(self):
        s=ttk.Style()
        theme="clam" if "clam" in s.theme_names() else s.theme_use()
        s.theme_use(theme)
        s.configure("TButton", padding=6)
        s.configure("Header.TLabel", font=("Segoe UI", 15, "bold"))
        s.configure("Caption.TLabel", foreground="#777")

    def _build_ui(self):
        top = ttk.Frame(self.root); top.pack(fill="x", padx=12, pady=8)
        ttk.Label(top, text="SATB 四声部编曲器", style="Header.TLabel").pack(side="left")
        ttk.Label(top, text="  OOP · 更好听的声音 · 简化操作", style="Caption.TLabel").pack(side="left")

        bar = ttk.Frame(self.root); bar.pack(fill="x", padx=12)
        ttk.Label(bar, text="BPM").pack(side="left")
        ttk.Spinbox(bar, from_=30, to=240, width=5, textvariable=self.bpm).pack(side="left", padx=(4,12))
        ttk.Label(bar, text="移调(半音)").pack(side="left")
        ttk.Spinbox(bar, from_=-24, to=24, width=5, textvariable=self.transpose).pack(side="left", padx=(4,12))

        ttk.Label(bar, text="音频后端").pack(side="left", padx=(8,2))
        cb=ttk.Combobox(bar, state="readonly", width=22, textvariable=self.backend_name,
                        values=["SoundFont (FluidSynth)","Sampler (WAV)","System MIDI"])
        cb.pack(side="left")

        ttk.Button(bar, text="选择 SoundFont", command=self.pick_sf2).pack(side="left", padx=6)
        ttk.Button(bar, text="选择采样目录", command=self.pick_sample_dir).pack(side="left", padx=6)
        ttk.Button(bar, text="▶ 播放", command=self.play).pack(side="left", padx=(20,6))
        ttk.Button(bar, text="■ 停止", command=self.stop).pack(side="left", padx=6)
        ttk.Button(bar, text="💾 导出 MIDI", command=self.export_midi).pack(side="left", padx=6)

        self.status = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x", padx=12, pady=(2,6))

        mid = ttk.Frame(self.root); mid.pack(fill="both", expand=True, padx=12, pady=6)
        self.texts={}
        for v in VOICE_ORDER:
            row=ttk.Frame(mid); row.pack(fill="x", pady=5)
            ttk.Label(row, text=f"{v}").pack(side="left")
            t=tk.Text(row, height=4, wrap="word"); t.pack(side="left", fill="x", expand=True)
            self.texts[v]=t

        gen = ttk.Labelframe(self.root, text="从和弦自动生成（基础声部连贯）")
        gen.pack(fill="x", padx=12, pady=8)
        self.chords = tk.StringVar(value="C F G7 C | Am Dm G7 C")
        self.durs   = tk.StringVar(value="q q q q q q q q")
        ttk.Label(gen, text="和弦：").grid(row=0,column=0,sticky="w")
        ttk.Entry(gen, textvariable=self.chords).grid(row=0,column=1,sticky="we", padx=6)
        ttk.Label(gen, text="时值：").grid(row=1,column=0,sticky="w")
        ttk.Entry(gen, textvariable=self.durs).grid(row=1,column=1,sticky="we", padx=6)
        btns=ttk.Frame(gen); btns.grid(row=0,column=2,rowspan=2, padx=6)
        ttk.Button(btns, text="覆盖填入", command=self.gen_chords_overwrite).pack(pady=2)
        ttk.Button(btns, text="在尾部追加", command=lambda:self.gen_chords_overwrite(append=True)).pack(pady=2)
        gen.columnconfigure(1, weight=1)

    def _load_demo(self):
        demo = {
            "S":"E4q F4q G4h | A4q G4q F4h | E4q E4q F4q G4q A4h",
            "A":"C4q D4q E4h | F4q E4q D4h | C4q C4q D4q E4q F4h",
            "T":"G3q A3q B3h | C4q B3q A3h | G3q G3q A3q B3q C4h",
            "B":"C3q F3q G3h | A2q G2q C3h | C3q C3q F2q G2q C3h"
        }
        for v in VOICE_ORDER:
            self.texts[v].delete("1.0","end")
            self.texts[v].insert("1.0", demo[v])

    # ----- 数据收集 -----
    def _collect_score(self):
        sc=Score(self.bpm.get())
        for v in VOICE_ORDER:
            text=self.texts[v].get("1.0","end").replace("|"," ")
            toks=[t for t in text.split() if t.strip()]
            seq=[]
            for t in toks:
                p,b=parse_token(t)
                seq.append(Note(p,b))
            sc.parts[v].extend(seq)
        sc.ensure_aligned()
        return sc

    # ----- 后端管理 -----
    def _build_backend(self):
        chosen=self.backend_name.get()
        if chosen.startswith("SoundFont"):
            be=FluidSynthBackend()
            be.load_soundfont(self.sf2_path.get())
            return be
        elif chosen.startswith("Sampler"):
            be=SamplerBackend()
            if self.sample_dir.get():
                n=be.load_folder(self.sample_dir.get())
                if n==0:
                    messagebox.showwarning("采样器","在该目录未找到如 piano_C4.wav 的样本；将改为 System MIDI 兜底。")
            return be
        else:
            return SystemMIDIVanilla()

    # ----- 播放导出 -----
    def play(self):
        # 关键：先清理旧状态，避免“首播要先点停止”
        self.stop()

        try:
            sc=self._collect_score()
        except Exception as e:
            messagebox.showerror("解析错误", str(e)); return
        events = score_to_events(sc, transpose=self.transpose.get())

        be=self._build_backend()
        try:
            be.prepare()
        except Exception as e:
            # 自动降级：SoundFont/采样器失败 → 尝试 System MIDI
            if not isinstance(be, SystemMIDIVanilla):
                try:
                    fallback=SystemMIDIVanilla(); fallback.prepare()
                    be=fallback
                    self.backend_name.set("System MIDI")
                    self.status.set(f"后端失败，已降级为 System MIDI：{e}")
                except Exception as e2:
                    messagebox.showerror("无法播放", f"{e}\n且兜底后端也不可用：{e2}")
                    return
            else:
                messagebox.showerror("无法播放", str(e)); return

        self.backend=be
        self.backend.start(events)
        self.playing=True
        # 提示所用后端/驱动
        if isinstance(be, FluidSynthBackend):
            self.status.set(f"播放中（{sc.bpm} BPM, 移调 {self.transpose.get()}） | SoundFont ✓ 驱动: {be.used_driver} 预设: {be.used_preset}")
        else:
            self.status.set(f"播放中（{sc.bpm} BPM, 移调 {self.transpose.get()}） | 后端: {be.name}")

    def stop(self):
        if self.backend:
            try: self.backend.stop()
            except: pass
        self.backend=None
        self.playing=False
        self.status.set("已停止。")

    def export_midi(self):
        try:
            sc=self._collect_score()
        except Exception as e:
            messagebox.showerror("解析错误", str(e)); return
        fn=filedialog.asksaveasfilename(defaultextension=".mid",
                filetypes=[("MIDI File","*.mid")], initialfile="satb.mid")
        if not fn: return
        mf=MIDIFile(4)
        for i,v in enumerate(VOICE_ORDER):
            mf.addTempo(i, 0, sc.bpm)
            mf.addProgramChange(i, VOICE_TO_CH[v], 0, 0)  # 全钢琴，避免花里胡哨
        for v in VOICE_ORDER:
            t=0.0
            for n in sc.parts[v].notes:
                if n.pitch is not None:
                    p=n.pitch+self.transpose.get()
                    mf.addNote(VOICE_TO_CH[v], VOICE_TO_CH[v], p, t, n.beats, n.vel)
                t+=n.beats
        with open(fn,"wb") as f: mf.writeFile(f)
        self.status.set(f"已导出 MIDI：{os.path.basename(fn)}")

    # ----- 和弦生成 -----
    def gen_chords_overwrite(self, append=False):
        prog=self.chords.get().replace("|"," ")
        chords=[c for c in prog.split() if c.strip()]
        durs=[d for d in self.durs.get().split() if d.strip()]
        try: parts=Arranger.chord_prog_to_satb(chords, durs)
        except Exception as e:
            messagebox.showerror("生成失败", str(e)); return
        def seq_to_text(seq):
            out=[]
            for n in seq:
                if n.pitch is None: out.append("Rq")
                else: out.append(f"{format_pitch(n.pitch)}q")
            return " ".join(out)
        for v in VOICE_ORDER:
            txt=seq_to_text(parts[v])
            if append:
                self.texts[v].insert("end", " | "+txt)
            else:
                self.texts[v].delete("1.0","end"); self.texts[v].insert("1.0", txt)
        self.status.set("已从和弦生成四声部。")

    # ----- 选择资源 -----
    def pick_sf2(self):
        path=filedialog.askopenfilename(filetypes=[("SoundFont","*.sf2"),("All files","*.*")])
        if path: self.sf2_path.set(path); self.status.set(f"SoundFont: {os.path.basename(path)}")
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
