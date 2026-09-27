#!/usr/bin/env python3
import sys,re,json,math,random,argparse,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

WIDTH,HEIGHT,FPS=1280,720,60
SAMPLE_RATE=44100
NYQUIST=SAMPLE_RATE/2
FFT_SIZE,POINTS=4096,360
CENTER_X,CENTER_Y=WIDTH/2,HEIGHT/2+20
HEART_SCALE_X=HEART_SCALE_Y=245
MAX_AMPLITUDE_PX=115.0
SPECTRUM_WIDTH=4
DB_FLOOR,DB_CEILING=-70.0,-10.0
SMOOTHING=0.72
MIN_LOG_FREQ=20.0
OKLCH_L,OKLCH_C=0.72,0.16
HUE_CYCLES_PER_SECOND=0.10
DEFAULT_FONT_PATH="/usr/share/fonts/office/bierstadt-d.ttf"
SUPPORTED_IMAGES={".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}
TRACK_TEXT_X,TRACK_TEXT_BOTTOM=40,40
ARTIST_FONT_SIZE,TITLE_FONT_SIZE=24,32
TRACK_TEXT_GAP=6
TEXT_COLOR=(245,245,245)
TEXT_STROKE_COLOR=(0,0,0)
TEXT_STROKE_WIDTH=2

def smoothstep(x):
    x=np.clip(x,0.0,1.0)
    return x*x*(3.0-2.0*x)

def clamp(x,lo,hi): return max(lo,min(hi,x))

def read_embedded_cuesheet(filename):
    cmd=["ffprobe","-v","error","-show_entries","format=duration:format_tags","-of","json",filename]
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if r.returncode: raise RuntimeError("ffprobe failed:\n"+r.stderr)
    fmt=json.loads(r.stdout).get("format",{})
    tags=fmt.get("tags",{})
    cue=next((v for k,v in tags.items() if k.lower()=="cuesheet"),None)
    if not cue: raise RuntimeError("No embedded CUESHEET tag found.")
    return cue,float(fmt.get("duration",0.0))

def cue_time_to_seconds(ts):
    m=re.fullmatch(r"(\d+):(\d+):(\d+)",ts.strip())
    if not m: raise ValueError(f"Invalid CUE timestamp: {ts}")
    minutes,seconds,frames=map(int,m.groups())
    return minutes*60.0+seconds+frames/75.0

def parse_cuesheet(cuesheet,audio_duration):
    tracks=[]; current=None
    for raw in cuesheet.splitlines():
        line=raw.strip()
        m=re.match(r"TRACK\s+(\d+)\s+AUDIO",line,re.I)
        if m:
            if current is not None: tracks.append(current)
            current={"number":int(m.group(1)),"title":"","performer":"","start":None}
            continue
        if current is None: continue
        m=re.match(r'TITLE\s+"(.*)"',line,re.I)
        if m: current["title"]=m.group(1); continue
        m=re.match(r'PERFORMER\s+"(.*)"',line,re.I)
        if m: current["performer"]=m.group(1); continue
        m=re.match(r"INDEX\s+01\s+(\d+:\d+:\d+)",line,re.I)
        if m: current["start"]=cue_time_to_seconds(m.group(1))
    if current is not None: tracks.append(current)
    tracks=sorted((t for t in tracks if t["start"] is not None),key=lambda t:t["start"])
    if not tracks: raise RuntimeError("No INDEX 01 entries found in cuesheet.")
    for i,t in enumerate(tracks):
        t["end"]=tracks[i+1]["start"] if i+1<len(tracks) else audio_duration
        t["duration"]=t["end"]-t["start"]
    return tracks

def find_track_at_time(tracks,t):
    for i in range(len(tracks)-1,-1,-1):
        if t>=tracks[i]["start"]: return i
    return 0

def scan_images(directory):
    d=Path(directory)
    if not d.is_dir(): raise RuntimeError(f"Image directory does not exist: {d}")
    images=sorted((p for p in d.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGES),key=lambda p:p.name.lower())
    if not images: raise RuntimeError(f"No supported images found in {d}")
    return images

def assign_images(images,count,rng):
    out=[]; pool=list(images); rng.shuffle(pool); pos=0
    while len(out)<count:
        if pos>=len(pool): pool=list(images); rng.shuffle(pool); pos=0
        out.append(pool[pos]); pos+=1
    return out

def prepare_image(path):
    with Image.open(path) as src:
        src=src.convert("RGB"); sw,sh=src.size
        scale=max(WIDTH/sw,HEIGHT/sh)
        rw=max(WIDTH,int(math.ceil(sw*scale))); rh=max(HEIGHT,int(math.ceil(sh*scale)))
        return src.resize((rw,rh),Image.Resampling.LANCZOS)

def track_pan_progress(t,track):
    if track["duration"]<=0: return 0.5
    p=clamp((t-track["start"])/track["duration"],0.0,1.0)
    return float(smoothstep((p-0.05)/0.90))

def crop_image_for_frame(img,p):
    iw,ih=img.size; ex=max(0,iw-WIDTH); ey=max(0,ih-HEIGHT)
    x=int(round(ex*p)); y=int(round(ey*p))
    return img.crop((x,y,x+WIDTH,y+HEIGHT))

def linear_to_srgb(x): return 12.92*x if x<=0.0031308 else 1.055*(x**(1/2.4))-0.055

def oklch_to_rgb(L,C,h):
    a=C*math.cos(h*2*math.pi); b=C*math.sin(h*2*math.pi)
    l_=L+0.3963377774*a+0.2158037573*b
    m_=L-0.1055613458*a-0.0638541728*b
    s_=L-0.0894841775*a-1.2914855480*b
    l,m,s=l_**3,m_**3,s_**3
    rgb=(4.0767416621*l-3.3077115913*m+0.2309699292*s,
         -1.2684380046*l+2.6097574011*m-0.3413193965*s,
         -0.0041960863*l-0.7034186147*m+1.7076147010*s)
    return tuple(int(round(np.clip(linear_to_srgb(v),0,1)*255)) for v in rgb)

def spectrum_colors(t,lm,rm):
    return (oklch_to_rgb(OKLCH_L,OKLCH_C,(t*HUE_CYCLES_PER_SECOND*lm)%1),
            oklch_to_rgb(OKLCH_L,OKLCH_C,(t*HUE_CYCLES_PER_SECOND*rm)%1))

def standard_heart(theta):
    x=16*math.sin(theta)**3
    y=13*math.cos(theta)-5*math.cos(2*theta)-2*math.cos(3*theta)-math.cos(4*theta)
    return x/17,y/17

def heart_point(side,u):
    x,y=standard_heart(math.pi+u*math.pi)
    return np.array([CENTER_X+abs(x)*side*HEART_SCALE_X,CENTER_Y-y*HEART_SCALE_Y],dtype=np.float64)

def make_half_heart(side):
    u=np.linspace(0,1,5000); pts=np.array([heart_point(side,x) for x in u])
    lengths=np.linalg.norm(np.diff(pts,axis=0),axis=1)
    cumulative=np.concatenate(([0.0],np.cumsum(lengths))); cumulative/=cumulative[-1]
    desired=np.linspace(0,1,POINTS); out=np.zeros((POINTS,2),dtype=np.float64)
    out[:,0]=np.interp(desired,cumulative,pts[:,0]); out[:,1]=np.interp(desired,cumulative,pts[:,1])
    return out

LEFT_BASE,RIGHT_BASE=make_half_heart(-1),make_half_heart(1)

def calculate_normals(points):
    out=np.zeros_like(points); center=np.array([CENTER_X,CENTER_Y])
    for i in range(len(points)):
        tangent=points[1]-points[0] if i==0 else points[-1]-points[-2] if i==len(points)-1 else points[i+1]-points[i-1]
        length=np.linalg.norm(tangent)
        if not length: continue
        tangent/=length; normal=np.array([-tangent[1],tangent[0]])
        if np.dot(normal,points[i]-center)<0: normal*=-1
        out[i]=normal
    return out

LEFT_NORMALS,RIGHT_NORMALS=calculate_normals(LEFT_BASE),calculate_normals(RIGHT_BASE)

def frequency_for_position(u): return 0.0 if u<=0 else MIN_LOG_FREQ*(NYQUIST/MIN_LOG_FREQ)**u
HEART_FREQUENCIES=np.array([frequency_for_position(u) for u in np.linspace(0,1,POINTS)])

def start_audio_decoder(filename,start,duration):
    cmd=["ffmpeg","-hide_banner","-loglevel","error","-ss",str(start),"-t",str(duration),"-i",filename,
         "-vn","-f","f32le","-acodec","pcm_f32le","-ac","2","-ar",str(SAMPLE_RATE),"-"]
    return subprocess.Popen(cmd,stdout=subprocess.PIPE)

def read_audio(process):
    raw=process.stdout.read(); process.stdout.close(); rc=process.wait()
    if rc: raise RuntimeError(f"FFmpeg analysis decoder exited with status {rc}")
    samples=np.frombuffer(raw,dtype="<f4")
    if len(samples)%2: samples=samples[:-1]
    return samples.reshape((-1,2))

WINDOW=np.hanning(FFT_SIZE).astype(np.float32)
FFT_FREQUENCIES=np.fft.rfftfreq(FFT_SIZE,d=1.0/SAMPLE_RATE)

def spectrum_for_frame(samples,frame_number):
    center=int(frame_number*SAMPLE_RATE/FPS); start=center-FFT_SIZE//2; end=start+FFT_SIZE
    block=np.zeros((FFT_SIZE,2),dtype=np.float32)
    a=max(0,start); b=min(len(samples),end)
    if b>a:
        da=a-start; block[da:da+b-a]=samples[a:b]
    block*=WINDOW[:,None]; norm=np.sum(WINDOW)/2
    return np.abs(np.fft.rfft(block[:,0]))/norm,np.abs(np.fft.rfft(block[:,1]))/norm

def magnitude_to_heart(fft):
    db=20*np.log10(np.maximum(fft,1e-12))
    mag=np.clip((np.interp(HEART_FREQUENCIES,FFT_FREQUENCIES,db)-DB_FLOOR)/(DB_CEILING-DB_FLOOR),0,1)
    mag[0]=mag[-1]=0; return mag

def draw_heart(image,left,right,t,lm,rm):
    draw=ImageDraw.Draw(image); lc,rc=spectrum_colors(t,lm,rm)
    lp=LEFT_BASE+LEFT_NORMALS*(left[:,None]*MAX_AMPLITUDE_PX)
    rp=RIGHT_BASE+RIGHT_NORMALS*(right[:,None]*MAX_AMPLITUDE_PX)
    points=lambda p:[(int(round(x)),int(round(y))) for x,y in p]
    draw.line(points(lp),fill=lc,width=SPECTRUM_WIDTH,joint="curve")
    draw.line(points(rp),fill=rc,width=SPECTRUM_WIDTH,joint="curve")

def track_text_alpha(t,track,hold,fade):
    e=t-track["start"]
    if e<0 or e>=track["duration"]: return 0.0
    if fade<=0: return 1.0 if e<hold else 0.0
    if e<fade: return float(smoothstep(e/fade))
    if e<fade+hold: return 1.0
    if e<fade+hold+fade: return float(1-smoothstep((e-fade-hold)/fade))
    return 0.0

def draw_track_info(image,track,artist_font,title_font,t,hold,fade):
    alpha=track_text_alpha(t,track,hold,fade)
    if alpha<=0: return
    artist=track["performer"].strip(); title=track["title"].strip()
    if not artist and not title: return
    overlay=Image.new("RGBA",image.size,(0,0,0,0)); draw=ImageDraw.Draw(overlay)
    ab=draw.textbbox((0,0),artist,font=artist_font,stroke_width=TEXT_STROKE_WIDTH) if artist else (0,0,0,0)
    tb=draw.textbbox((0,0),title,font=title_font,stroke_width=TEXT_STROKE_WIDTH) if title else (0,0,0,0)
    ah=ab[3]-ab[1]; th=tb[3]-tb[1]; y=HEIGHT-TRACK_TEXT_BOTTOM-ah-(TRACK_TEXT_GAP if artist and title else 0)-th
    fill=(*TEXT_COLOR,int(round(255*alpha))); stroke=(*TEXT_STROKE_COLOR,int(round(255*alpha)))
    if artist:
        draw.text((TRACK_TEXT_X,y),artist,font=artist_font,fill=fill,stroke_width=TEXT_STROKE_WIDTH,stroke_fill=stroke)
        y+=ah+(TRACK_TEXT_GAP if title else 0)
    if title: draw.text((TRACK_TEXT_X,y),title,font=title_font,fill=fill,stroke_width=TEXT_STROKE_WIDTH,stroke_fill=stroke)
    image.paste(Image.alpha_composite(image.convert("RGBA"),overlay).convert("RGB"))

def start_video_encoder(output,audio,start,duration):
    cmd=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pixel_format","rgb24",
         "-video_size",f"{WIDTH}x{HEIGHT}","-framerate",str(FPS),"-i","-","-ss",str(start),"-t",str(duration),"-i",audio,
         "-map","0:v:0","-map","1:a:0","-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p",
         "-c:a","flac","-sample_fmt","s32","-bits_per_raw_sample","24","-shortest",output]
    return subprocess.Popen(cmd,stdin=subprocess.PIPE)

def main():
    p=argparse.ArgumentParser(description="Render shuffled artwork plus a stereo heart-spectrum visualizer.")
    p.add_argument("input",help="input audio file")
    p.add_argument("-o","--output",default="heart-test.mkv")
    p.add_argument("--start",type=float,default=0.0)
    p.add_argument("--duration",type=float,default=30.0)
    p.add_argument("--images",required=True,help="directory containing track artwork")
    p.add_argument("--seed",type=int,default=None,help="reproduce a previous artwork shuffle")
    p.add_argument("--font",default=DEFAULT_FONT_PATH,help=f"track-info font (default: {DEFAULT_FONT_PATH})")
    p.add_argument("--text-hold",type=float,default=8.0)
    p.add_argument("--text-fade",type=float,default=1.5)
    a=p.parse_args()
    if a.start<0: p.error("--start cannot be negative")
    if a.duration<=0: p.error("--duration must be positive")
    if a.text_hold<0 or a.text_fade<0: p.error("text timing cannot be negative")
    try:
        artist_font=ImageFont.truetype(a.font,ARTIST_FONT_SIZE); title_font=ImageFont.truetype(a.font,TITLE_FONT_SIZE)
    except OSError as exc: raise RuntimeError(f"Could not load font: {a.font}") from exc
    seed=random.SystemRandom().randrange(0,2**63) if a.seed is None else a.seed
    rng=random.Random(seed); lm,rm=rng.random(),rng.random()
    print(f"Image shuffle seed: {seed}",file=sys.stderr)
    print(f"Hue multipliers: L={lm:.8f}, R={rm:.8f}",file=sys.stderr)
    cue,source_duration=read_embedded_cuesheet(a.input); tracks=parse_cuesheet(cue,source_duration)
    images=scan_images(a.images); assignments=assign_images(images,len(tracks),rng)
    available=max(0.0,source_duration-a.start); requested=min(a.duration,available)
    if requested<=0: raise RuntimeError("Requested start time is beyond EOF.")
    samples=read_audio(start_audio_decoder(a.input,a.start,requested))
    render_duration=min(requested,len(samples)/SAMPLE_RATE); total_frames=int(math.ceil(render_duration*FPS))
    encoder=start_video_encoder(a.output,a.input,a.start,render_duration)
    prev_l=np.zeros(POINTS); prev_r=np.zeros(POINTS); cached_i=None; cached_img=None
    try:
        for frame_number in range(total_frames):
            rel=frame_number/FPS; absolute=a.start+rel; i=find_track_at_time(tracks,absolute); track=tracks[i]
            if i!=cached_i:
                cached_img=prepare_image(assignments[i]); cached_i=i
            frame=crop_image_for_frame(cached_img,track_pan_progress(absolute,track))
            lf,rf=spectrum_for_frame(samples,frame_number)
            left=magnitude_to_heart(lf); right=magnitude_to_heart(rf)
            left=SMOOTHING*prev_l+(1-SMOOTHING)*left; right=SMOOTHING*prev_r+(1-SMOOTHING)*right
            left[0]=left[-1]=right[0]=right[-1]=0; prev_l,prev_r=left,right
            draw_heart(frame,left,right,absolute,lm,rm)
            draw_track_info(frame,track,artist_font,title_font,absolute,a.text_hold,a.text_fade)
            encoder.stdin.write(frame.tobytes())
            if frame_number%FPS==0 or frame_number==total_frames-1:
                print(f"\rRendering: {rel:7.1f} / {render_duration:.1f}s ({100*(frame_number+1)/total_frames:5.1f}%)",
                      end="",file=sys.stderr,flush=True)
    except BrokenPipeError:
        encoder.wait(); raise RuntimeError("FFmpeg stopped accepting video frames.")
    finally:
        if encoder.stdin:
            try: encoder.stdin.close()
            except BrokenPipeError: pass
    rc=encoder.wait(); print(file=sys.stderr)
    if rc: raise RuntimeError(f"FFmpeg encoder exited with status {rc}")
    print(f"Wrote {a.output}",file=sys.stderr)

if __name__=="__main__": main()
