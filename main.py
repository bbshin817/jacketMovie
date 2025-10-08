# lyricVideo.py
import sys, os, re, json, math, subprocess, shutil, tempfile, time
from typing import Tuple, List, Dict, Optional
import numpy as np
from PIL import Image, ImageStat
import moderngl
import pyglet
from pyglet import gl as pyg_gl
import freetype

# =========================
# スタイル（既定値／1920x1080基準）
# =========================
STYLE_DEFAULT = {
    "output_fps": 30,
    "internal_render_fps": 30,

    # パディング（解像度に相対：左右=100, 上下=150）
    "pad_x_base": 100,   # ←→
    "pad_y_base": 120,   # ↑↓

    # 情報表示（解像度に相対）
    "info_gap_x_base": 48,      # ジャケットとテキストの水平ギャップ
    "info_line_gap_base": 50,   # track と artist の縦ギャップ
    "info_title_line_height_base": 170,
    "info_artist_line_height_base": 67,
    "info_title_wrap_gap_base": 16,  # trackName 折返し行間

    # フォント（固定px）
    "track_font_px": 170,
    "artist_font_px": 50,
    "lyric_font_px": 120,

    # 歌詞レイアウト（固定px）
    "lyric_line_height": 150,
    "lyric_intra_row_gap": 10,    # 同一 words 内の折返し行間
    "lyric_inter_line_gap": 75,  # 別 words 間の行間

    # ジャケット
    "jacket_corner_radius_px": 20,
    "jacket_side_ratio": 256.0 / 1080.0,  # 1920x1080で256px相当

    # ディゾルブ
    "fade_sec": 1.0,              # 冒頭：情報→歌詞（ease-in-out）
    "final_wait_sec": 1.0,        # 最終行 end の1秒後に情報表示
    "final_fade_sec": 1.0,        # 情報表示のディゾルブ 1秒

    # 最終行のフェードアウト（最終行 end 到達時に開始）
    "last_line_fade_sec": 1.0,

    # 文節グラデーション境界の半幅（ピクセル）← 統一幅
    "syllable_edge_blur_px": 75.0,

    # NVENC
    "nvenc_preset": "p1",
    "nvenc_params": [
        "-pix_fmt", "yuv420p",
        "-rc", "cbr",
        "-b:v", "10M",
        "-maxrate", "10M",
        "-bufsize", "20M",
        "-rc-lookahead", "0",
        "-look_ahead", "0",
        "-bf", "0",
        "-spatial_aq", "0",
        "-temporal_aq", "0",
        "-g", "120",
        "-movflags", "+faststart",
        "-loglevel", "error"
    ],

    # box-shadow 近似
    "shadow_blur_px": 50.0,
    "shadow_spread_px": 1,
    "shadow_alpha": 0.25,
    
    "audio_delay" : 0.3
}

# プロジェクト直下の "fonts" ディレクトリを優先して探索
BASE_DIR = os.path.dirname(__file__)
FONT_DIR = os.path.join(BASE_DIR, "fonts")

# =========================
# ユーティリティ
# =========================

def _pick_font(candidates: list[str]) -> Optional[str]:
    for name in candidates:
        # まずは /fonts 内を探索
        p = os.path.join(FONT_DIR, name)
        q = ensure_ttf(p)
        if q:
            return q
        # 絶対パスや既存の相対指定が来た場合にも一応試す
        q = ensure_ttf(name)
        if q:
            return q
    return None

def parse_resolution(text: str) -> Tuple[int, int]:
    if not text:
        return 1920, 1080
    m = re.match(r"^(\d+)[xX](\d+)$", text)
    return (1920,1080) if not m else (int(m.group(1)), int(m.group(2)))

def clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))

def safe_filename(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]+", "_", name).strip() or "output"

def avg_color_with_clamp(jacket_path: str) -> Tuple[int,int,int]:
    img = Image.open(jacket_path).convert("RGB")
    stat = ImageStat.Stat(img)
    r, g, b = stat.mean
    return clamp_int(r,50,230), clamp_int(g,50,230), clamp_int(b,50,230)

def cubic_bezier(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    def bez(u, a, b): return 3*(1-u)*(1-u)*u*a + 3*(1-u)*u*u*b + u*u*u
    lo, hi = 0.0, 1.0
    x = max(0.0, min(1.0, x))
    for _ in range(20):
        mid = (lo + hi)/2
        if bez(mid, x1, x2) < x: lo = mid
        else: hi = mid
    u = (lo + hi)/2
    return bez(u, y1, y2)

def ease(x: float) -> float:
    return cubic_bezier(x, 0.25, 0.1, 0.25, 1.0)

def ease_in_out(x: float) -> float:
    return cubic_bezier(x, 0.42, 0.0, 0.58, 1.0)

def is_ascii_like(ch: str) -> bool:
    return bool(re.match(r"^[ -~]$", ch))

def ensure_ttf(font_path: str) -> Optional[str]:
    if not os.path.isfile(font_path):
        return None
    p = font_path.lower()
    if p.endswith(".ttf") or p.endswith(".otf"):
        return font_path
    if p.endswith(".woff2"):
        try:
            from fontTools.ttLib import TTFont
            import brotli  # noqa
            out_dir = os.path.join(tempfile.gettempdir(), "font_cache_gl")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, safe_filename(os.path.basename(font_path)) + ".ttf")
            if not os.path.isfile(out_path):
                tt = TTFont(font_path)
                tt.flavor = None
                tt.save(out_path)
            return out_path
        except Exception as e:
            print(f"[warn] WOFF2変換失敗: {e}")
            return None
    return None

def ffprobe_duration_seconds(path: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobeが見つかりません。PATHをご確認ください。")
    out = subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ]).decode().strip()
    try:
        return float(out)
    except Exception:
        return 0.0

# =========================
# words単位の時間整形（重なり解消）
# =========================
def normalize_words_timing(lines: List[dict]) -> None:
    """
    現在の行 endTimeMs > 次の行 startTimeMs の場合、
    現在行の endTimeMs を「次の行の endTimeMs」に置換します（words単位）。
    ※ in-place に書き換えます。
    """
    n = len(lines)
    for i in range(n - 1):
        cur_end = int(lines[i].get("endTimeMs", 0))
        nxt_start = int(lines[i+1].get("startTimeMs", 0))
        if cur_end > nxt_start:
            nxt_end = int(lines[i+1].get("endTimeMs", cur_end))
            lines[i]["endTimeMs"] = nxt_end

# =========================
# フォントアトラス
# =========================
class GlyphInfo:
    __slots__ = ("u0","v0","u1","v1","w","h","bearing_x","bearing_y","advance")
    def __init__(self,u0,v0,u1,v1,w,h,bx,by,adv):
        self.u0,self.v0,self.u1,self.v1 = u0,v0,u1,v1
        self.w,self.h = w,h
        self.bearing_x,self.bearing_y = bx,by
        self.advance = adv

class FontKey:
    __slots__=("path","size")
    def __init__(self,path: str,size: int):
        self.path=path; self.size=size
    def __hash__(self): return hash((self.path,self.size))
    def __eq__(self,o): return isinstance(o,FontKey) and o.path==self.path and o.size==self.size

class AtlasPacker:
    def __init__(self,width=4096,height=4096,padding=2):
        self.W,self.H=width,height
        self.pad=padding
        self.img=Image.new("L",(width,height),0)
        self.cx=self.pad; self.cy=self.pad; self.shelf_h=0
    def add_bitmap(self,bmp:Image.Image)->Tuple[int,int]:
        w,h=bmp.size
        if self.cx+w+self.pad>self.W:
            self.cx=self.pad
            self.cy+=self.shelf_h+self.pad
            self.shelf_h=0
        if h>self.shelf_h: self.shelf_h=h
        if self.cy+h+self.pad>self.H:
            raise RuntimeError("フォントアトラスが溢れました（4096x4096超）。")
        self.img.paste(bmp,(self.cx,self.cy))
        x,y=self.cx,self.cy
        self.cx+=w+self.pad
        return x,y

class FontAtlas:
    def __init__(self):
        self.glyphs: Dict[Tuple["FontKey",str], GlyphInfo] = {}
        self.packer = AtlasPacker()
        self.faces: Dict["FontKey", freetype.Face] = {}
    def _load_face(self,key:"FontKey")->freetype.Face:
        if key in self.faces: return self.faces[key]
        face=freetype.Face(key.path)
        face.set_char_size(key.size*64)
        self.faces[key]=face
        return face
    def add_text(self,key:"FontKey",text:str):
        face=self._load_face(key)
        for ch in text:
            k=(key,ch)
            if k in self.glyphs: continue
            face.load_char(ch, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
            glyph=face.glyph
            bmp=glyph.bitmap
            w,h=bmp.width,bmp.rows
            if w==0 or h==0:
                gi=GlyphInfo(0,0,0,0,0,0,glyph.bitmap_left,glyph.bitmap_top,glyph.advance.x/64.0)
                self.glyphs[k]=gi
                continue
            buf=np.array(bmp.buffer,dtype=np.uint8).reshape((h,w))
            im=Image.fromarray(buf)  # mode 省略（Deprecation回避）
            x,y=self.packer.add_bitmap(im)
            u0,v0,u1,v1 = x/self.packer.W, y/self.packer.H, (x+w)/self.packer.W, (y+h)/self.packer.H
            gi=GlyphInfo(u0,v0,u1,v1,w,h,glyph.bitmap_left,glyph.bitmap_top,glyph.advance.x/64.0)
            self.glyphs[k]=gi
    def finalize_gl(self,ctx:moderngl.Context)->moderngl.Texture:
        tex=ctx.texture((self.packer.W,self.packer.H),1,self.packer.img.tobytes())
        tex.filter=(moderngl.LINEAR,moderngl.LINEAR)
        tex.swizzle="RRRR"
        return tex

# =========================
# レイアウト構造
# =========================
class RenderRow:
    def __init__(self,char_indices:List[int],width:float):
        self.char_indices=char_indices
        self.width=width

class RenderLine:
    def __init__(self,idx:int,start:float,end:float,text:str,has_syll:bool):
        self.index=idx; self.start=start; self.end=end; self.text=text; self.has_syll=has_syll
        self.rows:List[RenderRow]=[]
        self.block_height:int=0
        self.char_times:List[Tuple[float,float]]=[]  # 各文字の (start,end)

# =========================
# Shaders（文節マスク連続グラデーション）
# =========================
TEXT_VS = """
#version 330
in vec2 a_pos;
in vec2 a_uv;
in float a_st;
in float a_et;
in float a_s0;     // 文節開始時刻
in float a_s1;     // 文節終了時刻
in float a_sx0;    // 文節 左端X（px）
in float a_sx1;    // 文節 右端X（px）
in float a_blur;   // 文節ぼかし半幅（正規化 0..0.5 程度）
in float a_has;    // 1.0 = syllableあり、0.0 = なし

uniform vec2 u_view;
uniform float u_time;
uniform float u_scroll;
uniform float u_alpha;
uniform vec2 u_offset;
uniform float u_shadow;

out vec2 v_uv;
out float v_px;
out float v_st;
out float v_et;
out float v_s0;
out float v_s1;
out float v_sx0;
out float v_sx1;
out float v_blur;
out float v_has;
out float v_alpha;

void main(){
    vec2 pos = a_pos + u_offset;
    pos.y -= u_scroll;

    v_uv = a_uv;
    v_px = pos.x;
    v_st = a_st;
    v_et = a_et;
    v_s0 = a_s0;
    v_s1 = a_s1;
    v_sx0 = a_sx0;
    v_sx1 = a_sx1;
    v_blur = a_blur;
    v_has = a_has;
    v_alpha = u_alpha;

    float x_ndc = (pos.x / u_view.x) * 2.0 - 1.0;
    float y_ndc = 1.0 - (pos.y / u_view.y) * 2.0;
    gl_Position = vec4(x_ndc, y_ndc, 0.0, 1.0);
}
"""

TEXT_FS = """
#version 330
uniform sampler2D u_tex;
uniform float u_shadow;
uniform float u_time;
uniform float u_static;  // 1.0 のとき常にアクティブ（情報表示など）

in vec2 v_uv;
in float v_px;
in float v_st;
in float v_et;
in float v_s0;
in float v_s1;
in float v_sx0;
in float v_sx1;
in float v_blur;
in float v_has;
in float v_alpha;

out vec4 fragColor;

float safe_norm(float x, float a, float b){
    float w = max(1e-6, b - a);
    return clamp((x - a)/w, 0.0, 1.0);
}

void main(){
    float a_glyph = texture(u_tex, v_uv).r;

    if(u_shadow > 0.5){
        float out_a = a_glyph * v_alpha;
        if(out_a <= 0.001) discard;
        fragColor = vec4(0.0, 0.0, 0.0, out_a);
        return;
    }

    float mixv = 1.0;

    if(u_static > 0.5){
        mixv = 1.0;
    }else if(v_has > 0.5){
        // syllables あり：文節マスクで左→右に白が広がる
        if(u_time <= v_s0){
            mixv = 0.0;
        }else if(u_time >= v_s1){
            mixv = 1.0;
        }else{
            // 時間 0..1 を境界中心とみなし、端では半幅を自動縮小
            float u = safe_norm(u_time, v_s0, v_s1);
            float nx = safe_norm(v_px, v_sx0, v_sx1);
            float halfspan = min(v_blur, min(u, 1.0 - u));
            float a = u - halfspan;
            float b = u + halfspan;
            // 左(白) → 右(黒) の境界を右へ移動
            mixv = 1.0 - smoothstep(a, b, nx);
        }
    }else{
        // syllables なし：行全体の0.5秒フェード（出現）
        float prog = 0.0;
        if(u_time <= v_st)      prog = 0.0;
        else if(u_time >= v_et) prog = 1.0;
        else                    prog = (u_time - v_st) / max(1e-6, v_et - v_st);
        mixv = clamp(prog, 0.0, 1.0);
    }

    vec3 rgb = mix(vec3(0.0), vec3(1.0), mixv);
    float alpha_scale = mix(0.5, 1.0, mixv);
    float out_a = a_glyph * v_alpha * alpha_scale;
    if(out_a <= 0.001) discard;
    fragColor = vec4(rgb, out_a);
}
"""

IMG_VS = """
#version 330
in vec2 a_pos;
in vec2 a_uv;
uniform vec2 u_view;
uniform float u_alpha;
out vec2 v_uv;
out float v_alpha;
void main(){
    v_uv = a_uv;
    v_alpha = u_alpha;
    float x_ndc = (a_pos.x / u_view.x) * 2.0 - 1.0;
    float y_ndc = 1.0 - (a_pos.y / u_view.y) * 2.0;
    gl_Position = vec4(x_ndc, y_ndc, 0.0, 1.0);
}
"""

IMG_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_size;
uniform float u_radius;
in vec2 v_uv;
in float v_alpha;
out vec4 fragColor;
void main(){
    vec4 c = texture(u_tex, v_uv);
    vec2 p = v_uv * u_size;
    float r = u_radius;

    bool outside =
        (p.x < r && p.y < r && length(p - vec2(r, r)) > r) ||
        (p.x > u_size.x - r && p.y < r && length(p - vec2(u_size.x - r, r)) > r) ||
        (p.x < r && p.y > u_size.y - r && length(p - vec2(r, u_size.y - r)) > r) ||
        (p.x > u_size.x - r && p.y > u_size.y - r && length(p - vec2(u_size.x - r, u_size.y - r)) > r);

    if(outside) discard;

    c.a *= v_alpha;
    if(c.a <= 0.001) discard;
    fragColor = c;
}
"""

# =========================
# 混在テキストの折り返し（英単語優先）
# =========================
_TOKEN_RE = re.compile(r"\s+|[A-Za-z0-9]+(?:[\'\-][A-Za-z0-9]+)*|.", re.UNICODE)

def wrap_mixed_by_width(text: str, content_w: float, advance_of) -> List[List[int]]:
    if text == "":
        return []
    advs = [advance_of(ch) for ch in text]
    rows: List[List[int]] = []
    cur: List[int] = []
    cur_w = 0.0

    tokens = list(_TOKEN_RE.finditer(text))
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        s = tok.group(0)
        a = tok.start()
        b = tok.end()

        if s.isspace():
            if cur:
                for j in range(a, b):
                    cur.append(j)
                    cur_w += advs[j]
            i += 1
            continue

        if re.fullmatch(r"[A-Za-z0-9]+(?:[\'\-][A-Za-z0-9]+)*", s):
            word_w = sum(advs[a:b])
            if cur and (cur_w + word_w) > content_w:
                while cur and text[cur[-1]] == " ":
                    cur_w -= advs[cur[-1]]
                    cur.pop()
                rows.append(cur)
                cur = []
                cur_w = 0.0
            if not cur and word_w > content_w:
                j = a
                while j < b:
                    w = advs[j]
                    if cur and (cur_w + w) > content_w:
                        rows.append(cur)
                        cur = []
                        cur_w = 0.0
                    cur.append(j)
                    cur_w += w
                    j += 1
            else:
                for j in range(a, b):
                    cur.append(j)
                    cur_w += advs[j]
            i += 1
            continue

        j = a
        while j < b:
            w = advs[j]
            if cur and (cur_w + w) > content_w:
                while cur and text[cur[-1]] == " ":
                    cur_w -= advs[cur[-1]]
                    cur.pop()
                rows.append(cur)
                cur = []
                cur_w = 0.0
            cur.append(j)
            cur_w += w
            j += 1
        i += 1

    if cur:
        while cur and text[cur[-1]] == " ":
            cur_w -= advs[cur[-1]]
            cur.pop()
        rows.append(cur)

    return rows

# =========================
# レイアウト（歌詞：混在折り返し + 文節情報）
# =========================
class RenderBuildResult:
    def __init__(self, lines: List[RenderLine], line_height: int):
        self.lines = lines
        self.line_height = line_height

def build_render_lines(
    lyric_json: List[dict],
    atlas: "FontAtlas",
    jp_key: "FontKey",
    lat_key: "FontKey",
    content_w: int,
    style: Dict[str, float]
) -> RenderBuildResult:

    lh = style["lyric_line_height"]
    gap_intra = style["lyric_intra_row_gap"]

    def char_key(c: str) -> "FontKey":
        return lat_key if is_ascii_like(c) else jp_key

    def char_advance(c: str) -> float:
        gi = atlas.glyphs.get((char_key(c), c))
        return gi.advance if gi else 0.0

    render_lines: List[RenderLine] = []

    for idx, item in enumerate(lyric_json):
        text = item.get("words", "")
        start = item.get("startTimeMs", 0)/1000.0
        end = item.get("endTimeMs", 0)/1000.0
        syllables = item.get("syllables", []) or []
        rl = RenderLine(idx, start, end, text, len(syllables)>0)

        # 文字ごとの時間
        if rl.has_syll and text != "":
            spans = []
            cur = 0
            for syl in syllables:
                n = int(syl.get("numChars", 0))
                s = syl.get("startTimeMs", item.get("startTimeMs", 0))/1000.0
                e = syl.get("endTimeMs", s)/1000.0
                spans.append((cur, cur+n, s, e))
                cur += n
            rl.char_times = []
            for i in range(len(text)):
                st, et = rl.start, rl.start + 0.5
                for a,b,s,e in spans:
                    if a <= i < b:
                        st, et = s, e
                        break
                rl.char_times.append((st, et))
        elif text != "":
            rl.char_times = [(start, start+0.5) for _ in range(len(text))]
        else:
            rl.char_times = []

        # 折り返し
        rows_idx = wrap_mixed_by_width(text, content_w, char_advance)
        rl.rows = []
        for idxs in rows_idx:
            w = sum(char_advance(text[i]) for i in idxs)
            rl.rows.append(RenderRow(idxs, w))

        # 高さ：空行は0、非空行は行送り×行数 + 行内ギャップ×(行数-1)
        if text == "":
            rl.block_height = 0
        else:
            rl.block_height = len(rl.rows)*lh + max(0, len(rl.rows)-1)*gap_intra

        render_lines.append(rl)

    return RenderBuildResult(render_lines, lh)

# =========================
# OpenGL Renderer
# =========================
class GLRenderer:
    def __init__(self, W:int, H:int, bg_rgb:Tuple[int,int,int], style:Dict[str, float]):
        self.W,self.H=W,H
        self.style = style
        config = pyglet.gl.Config(double_buffer=False, alpha_size=8, depth_size=0, stencil_size=0)
        self.window = pyglet.window.Window(width=W, height=H, visible=False, config=config)
        self.ctx = moderngl.create_context()
        self.fbo = self.ctx.simple_framebuffer((W,H))
        self.fbo.use()

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.prog_text = self.ctx.program(vertex_shader=TEXT_VS, fragment_shader=TEXT_FS)
        self.prog_img  = self.ctx.program(vertex_shader=IMG_VS,  fragment_shader=IMG_FS)

        self.prog_text["u_view"].value = (W,H)
        self.prog_text["u_time"].value = 0.0
        self.prog_text["u_scroll"].value = 0.0
        self.prog_text["u_alpha"].value = 1.0
        self.prog_text["u_offset"].value = (0.0, 0.0)
        self.prog_text["u_shadow"].value = 0.0
        self.prog_text["u_static"].value = 0.0

        self.prog_img["u_view"].value = (W,H)
        self.prog_img["u_alpha"].value = 1.0
        self.prog_img["u_size"].value = (0.0, 0.0)
        self.prog_img["u_radius"].value = 0.0

        self.atlas_tex = None
        self.text_vbo_info = None
        self.text_vbo_lyrics = None
        self.img_vbo = None
        self.jacket_tex = None

        # (first, count, row_top, row_bottom, is_last_row:int)
        self.lyric_ranges: List[Tuple[int,int,float,float,int]] = []

    def upload_atlas(self, atlas_tex: moderngl.Texture):
        self.atlas_tex = atlas_tex

    def build_text_geometry(
        self,
        atlas: "FontAtlas",
        items: List[dict],
        row_height_for_bounds: Optional[int]=None,
        record_ranges: bool=False,
        baseline_to_top_px: Optional[int]=None
    ) -> moderngl.Buffer:
        """
        items 要素:
          {
            "text": str,
            "x": int, "y": int (ベースライン),
            "font_keys": Optional[List[FontKey]] | "font_key": FontKey,
            "char_times": Optional[List[(st,et)]],

            # 文節グラデーション用（任意・無ければ0で描画）
            "syl_s0": Optional[List[float]],
            "syl_s1": Optional[List[float]],
            "syl_sx0": Optional[List[float]],
            "syl_sx1": Optional[List[float]],
            "syl_blur": Optional[List[float]],
            "syl_has": Optional[List[float]],

            # 最終行の行かどうか（行単位）
            "is_last_row": Optional[bool]
          }
        """
        verts=[]
        ranges=[]
        first_vertex = 0

        def glyph_of(k:"FontKey", ch:str) -> Optional[GlyphInfo]:
            return atlas.glyphs.get((k, ch))

        for it in items:
            x=float(it["x"]); y=float(it["y"])
            text=it["text"]
            times=it.get("char_times")

            s0s = it.get("syl_s0")
            s1s = it.get("syl_s1")
            sx0s= it.get("syl_sx0")
            sx1s= it.get("syl_sx1")
            sbl = it.get("syl_blur")
            shs = it.get("syl_has")

            font_keys: Optional[List["FontKey"]] = it.get("font_keys")
            single_key: Optional["FontKey"] = it.get("font_key")

            cx=x
            begin = first_vertex

            for i,ch in enumerate(text):
                k = font_keys[i] if font_keys else single_key
                if not k:
                    continue
                gi = glyph_of(k, ch)
                if not gi:
                    continue

                x0 = cx + gi.bearing_x
                y0 = y - gi.bearing_y
                x1 = x0 + gi.w
                y1 = y0 + gi.h
                u0,v0,u1,v1 = gi.u0,gi.v0,gi.u1,gi.v1

                st, et = (0.0, 0.0)
                if times and i < len(times):
                    st, et = times[i]

                ss0 = s0s[i] if s0s else 0.0
                ss1 = s1s[i] if s1s else 0.0
                xs0 = sx0s[i] if sx0s else 0.0
                xs1 = sx1s[i] if sx1s else 1.0
                sbr = sbl[i] if sbl else 0.0
                sh  = shs[i] if shs else 0.0

                for vx,vy,uu,vv in [(x0,y0,u0,v0),(x1,y0,u1,v0),(x1,y1,u1,v1),
                                    (x0,y0,u0,v0),(x1,y1,u1,v1),(x0,y1,u0,v1)]:
                    verts.extend([vx,vy, uu,vv, st,et, ss0,ss1, xs0,xs1, sbr,sh])
                cx += gi.advance
                first_vertex += 6

            count = first_vertex - begin
            if record_ranges and row_height_for_bounds is not None and baseline_to_top_px is not None:
                row_top = y - baseline_to_top_px
                row_bottom = row_top + row_height_for_bounds
                is_last = 1 if it.get("is_last_row", False) else 0
                ranges.append((begin, count, row_top, row_bottom, is_last))

        data = np.array(verts, dtype="f4").tobytes()
        vbo = self.ctx.buffer(data)
        if record_ranges:
            self.lyric_ranges = ranges
        return vbo

    def build_jacket_geometry(self, x:int, y:int, w:int, h:int)->moderngl.Buffer:
        x0,y0,x1,y1 = float(x),float(y),float(x+w),float(y+h)
        verts = [
            x0,y0, 0.0,0.0,
            x1,y0, 1.0,0.0,
            x1,y1, 1.0,1.0,
            x0,y0, 0.0,0.0,
            x1,y1, 1.0,1.0,
            x0,y1, 0.0,1.0
        ]
        return self.ctx.buffer(np.array(verts,dtype="f4").tobytes())

    def upload_jacket_image(self, jacket_path:str, side:int)->moderngl.Texture:
        img = Image.open(jacket_path).convert("RGBA").resize((side,side), Image.LANCZOS)
        tex = self.ctx.texture(img.size, 4, img.tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    def begin(self, bg_rgb:Tuple[int,int,int]):
        pyg_gl.glDisable(pyg_gl.GL_SCISSOR_TEST)
        self.fbo.use()
        self.ctx.clear(bg_rgb[0]/255.0, bg_rgb[1]/255.0, bg_rgb[2]/255.0, 1.0)

    def draw_text_full(self, vbo:moderngl.Buffer, alpha:float, t:float, scroll:float, static:bool, offset:Tuple[float,float]=(0.0,0.0), shadow:bool=False):
        vao = self.ctx.vertex_array(
            self.prog_text, [
                (vbo, "2f 2f 1f 1f 1f 1f 1f 1f 1f 1f",
                 "a_pos","a_uv","a_st","a_et","a_s0","a_s1","a_sx0","a_sx1","a_blur","a_has")
            ]
        )
        self.prog_text["u_time"].value = float(t)
        self.prog_text["u_scroll"].value = float(scroll)
        self.prog_text["u_alpha"].value = float(alpha)
        self.prog_text["u_offset"].value = (float(offset[0]), float(offset[1]))
        self.prog_text["u_shadow"].value = 1.0 if shadow else 0.0
        self.prog_text["u_static"].value = 1.0 if static else 0.0
        self.atlas_tex.use(0)
        self.prog_text["u_tex"].value = 0
        vao.render(moderngl.TRIANGLES)

    def draw_text_culled(self, vbo:moderngl.Buffer, base_alpha:float, t:float, scroll:float, y_clip_min:int, y_clip_max:int, overscan:int=200, last_row_alpha:float=1.0):
        if not self.lyric_ranges:
            self.draw_text_full(vbo, base_alpha, t, scroll, static=False)
            return
        vao = self.ctx.vertex_array(
            self.prog_text, [
                (vbo, "2f 2f 1f 1f 1f 1f 1f 1f 1f 1f",
                 "a_pos","a_uv","a_st","a_et","a_s0","a_s1","a_sx0","a_sx1","a_blur","a_has")
            ]
        )
        self.prog_text["u_time"].value = float(t)
        self.prog_text["u_scroll"].value = float(scroll)
        self.prog_text["u_offset"].value = (0.0, 0.0)
        self.prog_text["u_shadow"].value = 0.0
        self.prog_text["u_static"].value = 0.0
        self.atlas_tex.use(0)
        self.prog_text["u_tex"].value = 0

        y_min = y_clip_min - overscan
        y_max = y_clip_max + overscan

        for first, count, row_top, row_bottom, is_last in self.lyric_ranges:
            top = row_top - scroll
            bottom = row_bottom - scroll
            if bottom < y_min or top > y_max:
                continue
            a = base_alpha * (last_row_alpha if is_last == 1 else 1.0)
            self.prog_text["u_alpha"].value = float(a)
            vao.render(moderngl.TRIANGLES, vertices=count, first=first)

    def draw_image(self, vbo:moderngl.Buffer, tex:moderngl.Texture, alpha:float, side:int, corner_radius:float):
        vao = self.ctx.simple_vertex_array(self.prog_img, vbo, "a_pos","a_uv")
        self.prog_img["u_alpha"].value = float(alpha)
        self.prog_img["u_size"].value = (float(side), float(side))
        self.prog_img["u_radius"].value = float(corner_radius)
        tex.use(0)
        self.prog_img["u_tex"].value = 0
        vao.render(moderngl.TRIANGLES)

    def read_rgb(self)->bytes:
        data = self.fbo.read(components=3, alignment=1)
        arr = np.frombuffer(data, dtype=np.uint8).reshape(self.H, self.W, 3)
        arr = np.flip(arr, axis=0).copy()
        return arr.tobytes()

# =========================
# 影オフセット生成（box-shadow近似）
# =========================
def make_shadow_offsets(blur_px: float, spread_px: float) -> List[Tuple[float,float]]:
    offsets: List[Tuple[float,float]] = []
    ring_count = max(6, int(blur_px / 6.0))
    radii = [0.0]
    if spread_px > 0:
        radii.append(spread_px)
    for i in range(1, ring_count+1):
        r = (i / ring_count) * blur_px
        radii.append(r)
    for r in radii:
        if r == 0.0:
            offsets.append((0.0, 0.0))
            continue
        samples = max(8, int(2 * math.pi * r / 6.0))
        for k in range(samples):
            a = (2*math.pi) * (k / samples)
            offsets.append((math.cos(a) * r, math.sin(a) * r))
    return offsets

def build_scroll_events(render_lines: List["RenderLine"], inter_gap: int) -> List[Tuple[float, float]]:
    """
    各行は、自行の終了時刻に到達したときのみ単独でスクロールします。
    - イベントは (event_time, displacement_px)
    - displacement は「その行の block_height + inter_gap」
    - 最後の行はスクロールしない（イベントを発行しない）
    """
    n = len(render_lines)
    events: List[Tuple[float, float]] = []
    if n <= 1:
        return events

    last_idx = n - 1
    for i in range(0, last_idx):  # 最後の行は除外
        line = render_lines[i]
        disp = float(line.block_height + inter_gap)
        if disp > 0.0:
            events.append((line.end, disp))

    return events

# =========================
# 生成本体（モジュールAPI）
# =========================
def make(
    audio_path: str,
    lyric_path: str,
    jacket_path: str,
    info_path: str,
    resolution: str = "1920x1080",
    style_overrides: Optional[Dict[str, float]] = None
) -> str:
    STYLE = dict(STYLE_DEFAULT)
    if style_overrides:
        for k, v in style_overrides.items():
            if k in STYLE:
                STYLE[k] = v

    W, H = parse_resolution(resolution)

    with open(lyric_path, "r", encoding="utf-8") as f:
        lyric_data = json.load(f)

    # ★ words単位の時間整形（ご要望ロジック）
    normalize_words_timing(lyric_data)

    with open(info_path, "r", encoding="utf-8") as f:
        info_data = json.load(f)
    track = info_data.get("trackName","Unknown Track")
    artist = info_data.get("artistName","Unknown Artist")

    audio_sec = ffprobe_duration_seconds(audio_path)
    first_start_ms = min([ln.get("startTimeMs",0) for ln in lyric_data]) if lyric_data else 0
    first_start = first_start_ms/1000.0

    last_end_sec = max([ln.get("endTimeMs",0) for ln in lyric_data])/1000.0 if lyric_data else audio_sec

    # 冒頭の情報→歌詞
    t_in = 0.0 if first_start < 5.0 else max(0.0, first_start - 5.0)
    fade_in_dur = STYLE["fade_sec"]

    # 終端：最終行はスクロールしない + 最終行フェードアウト → 1秒後に情報表示を1秒でディゾルブ
    t_info_in = last_end_sec + STYLE["final_wait_sec"]
    info_fade_dur = STYLE["final_fade_sec"]
    last_fade_dur = STYLE["last_line_fade_sec"]

    # 全体尺（オーディオ末尾 + 安全マージン）
    total_sec = max(audio_sec + STYLE["audio_delay"] + 2.0, t_info_in + info_fade_dur)

    sx = W / 1920.0
    sy = H / 1080.0
    track_px = max(8, int(round(STYLE["track_font_px"]  * sx)))
    artist_px = max(8, int(round(STYLE["artist_font_px"] * sx)))
    lyric_px  = max(8, int(round(STYLE["lyric_font_px"]  * sx)))
    lyric_line_height_scaled   = int(round(STYLE["lyric_line_height"]   * sx))
    lyric_intra_row_gap_scaled = int(round(STYLE["lyric_intra_row_gap"] * sx))
    lyric_inter_line_gap_scaled= int(round(STYLE["lyric_inter_line_gap"]* sx))

    # STYLE に上書きして以降の処理に反映（レイアウト計算・カリング・スクロールなど全てに効く）
    STYLE["lyric_line_height"]    = lyric_line_height_scaled
    STYLE["lyric_intra_row_gap"]  = lyric_intra_row_gap_scaled
    STYLE["lyric_inter_line_gap"] = lyric_inter_line_gap_scaled
    pad_x = int(round(STYLE["pad_x_base"] * sx))
    pad_y = int(round(STYLE["pad_y_base"] * sy))
    info_gap_x = int(round(STYLE["info_gap_x_base"] * sx))
    info_line_gap = int(round(STYLE["info_line_gap_base"] * sy))
    info_title_lh = int(round(STYLE["info_title_line_height_base"] * sy))
    info_artist_lh = int(round(STYLE["info_artist_line_height_base"] * sy))
    info_title_wrap_gap = int(round(STYLE["info_title_wrap_gap_base"] * sy))

    bg = avg_color_with_clamp(jacket_path)

    info_font_path = _pick_font([
        "UDkakugo.woff2", "UDkakugo.ttf", "UDkakugo.otf", "UDKakugo-Regular.ttf"
    ]) or "DejaVuSans.ttf"

    jp_font_path = _pick_font([
        "hirakaku_w6.woff2", "hirakaku_w6.ttf", "HiraginoSans-W6.otf", "HiraginoKakuGothic-W6.otf"
    ]) or "DejaVuSans.ttf"

    lat_font_path = _pick_font([
        "GoogleSansMedium.woff2", "GoogleSans-Medium.ttf", "GoogleSans-Medium.otf", "GoogleSans.ttf"
    ]) or "DejaVuSans.ttf"

    atlas = FontAtlas()
    fk_info_title = FontKey(info_font_path, track_px)
    fk_info_artist= FontKey(info_font_path, artist_px)
    fk_lyric_jp   = FontKey(jp_font_path,   lyric_px)
    fk_lyric_lat  = FontKey(lat_font_path,  lyric_px)

    info_chars = set((track or "") + (artist or ""))
    lyric_chars = set("".join([ln.get("words","") for ln in lyric_data if ln.get("words","") != ""]))

    atlas.add_text(fk_info_title, "".join(info_chars))
    atlas.add_text(fk_info_artist, "".join(info_chars))
    atlas.add_text(fk_lyric_jp,  "".join([c for c in lyric_chars if not is_ascii_like(c)]))
    atlas.add_text(fk_lyric_lat, "".join([c for c in lyric_chars if is_ascii_like(c)]))

    content_w = W - pad_x*2
    content_h = H - pad_y*2

    # ===== 歌詞レイアウト =====
    rres = build_render_lines(
        lyric_json=lyric_data,
        atlas=atlas,
        jp_key=fk_lyric_jp,
        lat_key=fk_lyric_lat,
        content_w=content_w,
        style=STYLE
    )
    render_lines = rres.lines

    # ===== OpenGL 準備 =====
    renderer = GLRenderer(W,H,bg, STYLE)
    atlas_tex = atlas.finalize_gl(renderer.ctx)
    renderer.upload_atlas(atlas_tex)

    # ===== ジャケット =====
    side = int(round(min(W,H) * STYLE["jacket_side_ratio"]))
    jacket_tex = renderer.upload_jacket_image(jacket_path, side)

    # ===== trackName 折り返し（artist を侵食しない）=====
    def adv_title(ch: str) -> float:
        gi = atlas.glyphs.get((fk_info_title, ch))
        return gi.advance if gi else 0.0
    info_text_w = content_w - side - info_gap_x
    track = track or ""
    track_rows_idx = wrap_mixed_by_width(track, info_text_w, adv_title)

    # 情報表示のテキストボックス高さ
    track_block_h = (len(track_rows_idx) * info_title_lh + max(0, len(track_rows_idx)-1) * info_title_wrap_gap) if track_rows_idx else info_title_lh
    text_block_h = track_block_h + info_line_gap + info_artist_lh
    block_h = max(side, text_block_h)
    base_y = pad_y + (content_h - block_h)//2

    jacket_x = pad_x
    jacket_y = base_y + (block_h - side)//2
    text_box_top = base_y + (block_h - text_block_h)//2
    text_left_x = pad_x + side + info_gap_x

    # track の各行ベースライン
    title_baselines = []
    yb = text_box_top + track_px
    if track_rows_idx:
        for _ in range(len(track_rows_idx)):
            title_baselines.append(yb)
            yb += info_title_lh + info_title_wrap_gap
    else:
        title_baselines.append(yb)

    artist_baseline_y = text_box_top + track_block_h + info_line_gap + artist_px

    # ===== 情報表示テキスト VBO（track 複数行 + artist）=====
    info_items = []
    if track_rows_idx:
        for row_indices, baseline in zip(track_rows_idx, title_baselines):
            s = "".join([ track[i] for i in row_indices ])
            info_items.append({
                "text": s,
                "x": int(text_left_x),
                "y": int(baseline),
                "font_key": fk_info_title,
                "char_times": None
            })
    else:
        info_items.append({
            "text": track or " ",
            "x": int(text_left_x),
            "y": int(title_baselines[0]),
            "font_key": fk_info_title,
            "char_times": None
        })
    info_items.append({
        "text": artist or " ",
        "x": int(text_left_x),
        "y": int(artist_baseline_y),
        "font_key": fk_info_artist,
        "char_times": None
    })
    vbo_info = renderer.build_text_geometry(
        atlas, info_items,
        row_height_for_bounds=None, record_ranges=False, baseline_to_top_px=None
    )
    renderer.text_vbo_info = vbo_info
    renderer.img_vbo = renderer.build_jacket_geometry(jacket_x, jacket_y, side, side)
    renderer.jacket_tex = jacket_tex

    # 情報表示テキスト（シャドウ）
    shadow_offsets = make_shadow_offsets(STYLE["shadow_blur_px"], STYLE["shadow_spread_px"])
    shadow_taps = len(shadow_offsets)
    shadow_per_tap_alpha = STYLE["shadow_alpha"] / max(1, shadow_taps)

    # ===== 歌詞ジオメトリ構築（最終行フラグ埋め込み）=====
    lyric_items=[]
    y0 = pad_y
    blur_half_px = float(STYLE["syllable_edge_blur_px"])
    last_idx = len(render_lines)-1 if render_lines else -1

    for li, rl in enumerate(render_lines):
        row_y = y0
        if rl.rows:
            for row in rl.rows:
                s = "".join([rl.text[i] for i in row.char_indices])
                font_keys = [(fk_lyric_lat if is_ascii_like(ch) else fk_lyric_jp) for ch in s]
                times = [rl.char_times[i] for i in row.char_indices] if rl.char_times else [(rl.start, rl.start+0.5) for _ in row.char_indices]

                syl_s0=[]; syl_s1=[]; syl_sx0=[]; syl_sx1=[]; syl_blur=[]; syl_has=[]

                # 文字 advance と累積X
                advs=[]
                for ch, fk in zip(s, font_keys):
                    gi = atlas.glyphs.get((fk, ch))
                    advs.append(gi.advance if gi else 0.0)
                x_char=[0.0]
                for a in advs[:-1]:
                    x_char.append(x_char[-1] + a)

                if rl.has_syll and s:
                    g_start = 0
                    while g_start < len(times):
                        g_st, g_et = times[g_start]
                        g_end = g_start + 1
                        while g_end < len(times) and times[g_end] == (g_st, g_et):
                            g_end += 1
                        sx0 = float(pad_x) + x_char[g_start]
                        sx1 = float(pad_x) + x_char[g_end-1] + advs[g_end-1]
                        width_px = max(1.0, sx1 - sx0)
                        blur_norm = min(0.5, blur_half_px / width_px)
                        for j in range(g_start, g_end):
                            syl_s0.append(g_st)
                            syl_s1.append(g_et)
                            syl_sx0.append(sx0)
                            syl_sx1.append(sx1)
                            syl_blur.append(blur_norm)
                            syl_has.append(1.0)
                        g_start = g_end
                else:
                    n = len(s)
                    syl_s0 = [0.0]*n
                    syl_s1 = [0.0]*n
                    syl_sx0= [0.0]*n
                    syl_sx1= [1.0]*n
                    syl_blur=[0.0]*n
                    syl_has = [0.0]*n

                lyric_items.append({
                    "text": s,
                    "x": int(pad_x),
                    "y": int(row_y + lyric_px),  # baseline
                    "font_keys": font_keys,
                    "char_times": times,
                    "syl_s0": syl_s0,
                    "syl_s1": syl_s1,
                    "syl_sx0": syl_sx0,
                    "syl_sx1": syl_sx1,
                    "syl_blur": syl_blur,
                    "syl_has": syl_has,
                    "is_last_row": (li == last_idx)
                })

                row_y += STYLE["lyric_line_height"] + STYLE["lyric_intra_row_gap"]

        y0 += (rl.block_height if rl.rows else 0) + STYLE["lyric_inter_line_gap"]

    vbo_lyrics = renderer.build_text_geometry(
        atlas, lyric_items,
        row_height_for_bounds=STYLE["lyric_line_height"],
        record_ranges=True,
        baseline_to_top_px=lyric_px
    )
    renderer.text_vbo_lyrics = vbo_lyrics

    # ===== “重なりグループ終端待ち”スクロールイベント =====
    inter_gap = STYLE["lyric_inter_line_gap"]
    scroll_events = build_scroll_events(render_lines, inter_gap)

    def scroll_offset_grouped(t: float) -> float:
        off = 0.0
        for ev_t, disp in scroll_events:
            dt = t - ev_t
            if dt <= 0:
                continue
            if dt >= 0.5:
                off += disp
            else:
                off += disp * ease(dt/0.5)
        return off

    # ===== ディゾルブ係数（情報↔歌詞）=====
    def alphas(tt: float) -> Tuple[float,float]:
        # 冒頭
        if tt < t_in:
            return 1.0, 0.0
        if t_in <= tt < t_in + fade_in_dur:
            u = (tt - t_in)/fade_in_dur
            ue = ease_in_out(u)
            return 1.0 - ue, ue
        # 中間（歌詞メイン）
        if tt < t_info_in:
            return 0.0, 1.0
        # 終端：情報表示を1秒でディゾルブ（歌詞は個別制御：最終行のみフェード）
        if t_info_in <= tt < t_info_in + info_fade_dur:
            u = (tt - t_info_in)/info_fade_dur
            ue = ease_in_out(u)
            return ue, 1.0  # 歌詞はこの段階で全体αは1.0（最終行は別途フェード）
        return 1.0, 1.0

    # ===== 最終行のフェードアウト係数（行単体）=====
    def last_line_alpha(tt: float) -> float:
        if tt < last_end_sec:
            return 1.0
        u = (tt - last_end_sec) / max(1e-6, last_fade_dur)
        if u >= 1.0:
            return 0.0
        return 1.0 - ease_in_out(max(0.0, min(1.0, u)))

    # ===== ffmpeg（NVENC）=====
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpegが見つかりません。PATHをご確認ください。")
    out_name = f"{safe_filename(track)} - {safe_filename(artist)}.mp4"
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(STYLE["output_fps"]), "-i", "-",
        "-itsoffset", str(STYLE["audio_delay"]),
        "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "h264_nvenc", "-preset", STYLE["nvenc_preset"],
        *STYLE["nvenc_params"],
        "-c:a", "aac",
        out_name
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=1024*1024)

    total_frames_out = int(round(total_sec * STYLE["output_fps"]))
    fps_in = STYLE["internal_render_fps"]
    duplicate = max(1, int(round(STYLE["output_fps"] / fps_in)))
    frames_in_total = int(math.ceil(total_sec * fps_in))
    t_step = 1.0 / fps_in
    start_time = time.time()
    written_out = 0

    # ===== ループ =====
    t = 0.0
    for _ in range(frames_in_total):
        info_a, lyric_base_a = alphas(t)
        sc = scroll_offset_grouped(t)
        last_a = last_line_alpha(t)

        renderer.begin(bg)

        # 情報表示（影→本体）
        if info_a > 0.001:
            renderer.draw_image(renderer.img_vbo, renderer.jacket_tex, info_a, side, STYLE["jacket_corner_radius_px"])
            for dx,dy in shadow_offsets:
                renderer.draw_text_full(renderer.text_vbo_info, shadow_per_tap_alpha, 0.0, 0.0, static=True, offset=(dx,dy), shadow=True)
            renderer.draw_text_full(renderer.text_vbo_info, info_a, 0.0, 0.0, static=True)

        # 歌詞（全体αは lyric_base_a、最終行のみ last_a を掛け合わせる）
        if lyric_base_a > 0.001:
            renderer.draw_text_culled(
                renderer.text_vbo_lyrics, lyric_base_a, t, sc,
                y_clip_min=0, y_clip_max=H,
                overscan=STYLE["lyric_line_height"],
                last_row_alpha=last_a
            )

        rgb = renderer.read_rgb()
        for _dup in range(duplicate):
            if written_out >= total_frames_out:
                break
            proc.stdin.write(rgb)
            written_out += 1

        # 進捗表示
        elapsed = time.time() - start_time
        progress = written_out / max(1, total_frames_out)
        fps_est = (written_out / elapsed) if elapsed > 0 else 0.0
        eta = (total_frames_out - written_out) / fps_est if fps_est > 0 else 0.0
        print(f"\rProgress: {written_out}/{total_frames_out} frames ({progress*100:5.1f}%) | {fps_est:5.1f} fps | ETA {eta:6.1f}s", end="")

        if written_out >= total_frames_out:
            break
        t += t_step

    print("")
    proc.stdin.close()
    proc.wait()
    print(f"Done: {out_name}")
    return out_name

# =========================
# CLI
# =========================
def _cli():
    if len(sys.argv) < 5:
        print("Usage: python lyricVideo.py <audio.wav> <lyrics.json> <jacket.jpg> <info.json> [WxH]")
        sys.exit(1)
    audio_path = sys.argv[1]
    lyric_path = sys.argv[2]
    jacket_path = sys.argv[3]
    info_path = sys.argv[4]
    res = sys.argv[5] if len(sys.argv) >= 6 else "1920x1080"
    make(audio_path, lyric_path, jacket_path, info_path, res)

if __name__ == "__main__":
    _cli()
