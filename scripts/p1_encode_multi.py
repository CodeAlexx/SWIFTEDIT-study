# Encode the source + several EDIT prompts in ONE T5 session (amortize the ~25min fp32 load).
# Saves p1_embeds_multi.pt with {src, and one entry per edit} for FlowEdit generality demos.
# OOM-safe: encoder-only (~8GB bf16); run inside a MemoryMax-capped user scope.
import os, time, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import T5EncoderModel, AutoTokenizer

REPO="google/t5-v1_1-xxl"; DEV,DT="cuda",torch.bfloat16; MAXLEN=128
SRC = "a red sports car driving on a coastal road"
EDITS = {
    "blue":   "a blue sports car driving on a coastal road",           # color (control)
    "sunset": "a red sports car driving on a coastal road at sunset, dramatic orange sky",  # scene/light
    "truck":  "a yellow pickup truck driving on a coastal road",        # object swap
    "snow":   "a red sports car driving on a coastal road in heavy snow",  # weather
}
OUT="/home/alex/swiftedit-ltx-research/p1_embeds_multi.pt"

def main():
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(REPO)
    print("loading T5 encoder (bf16, encoder-only)...",flush=True)
    enc=T5EncoderModel.from_pretrained(REPO, torch_dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    print(f"  loaded in {time.time()-t0:.0f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB",flush=True)
    def embed(p):
        b=tok([p],max_length=MAXLEN,padding="max_length",truncation=True,return_tensors="pt")
        ids=b.input_ids.to(DEV); m=b.attention_mask.to(DEV)
        with torch.no_grad(): h=enc(ids,attention_mask=m).last_hidden_state
        return h.cpu(), m.cpu()
    out={"maxlen":MAXLEN,"src_prompt":SRC}
    hs,ms=embed(SRC); out["src"]=hs; out["src_mask"]=ms
    for k,p in EDITS.items():
        he,me=embed(p); out[k]=he; out[k+"_mask"]=me; out[k+"_prompt"]=p
        print(f"  encoded '{k}': {tuple(he.shape)}  <- {p}",flush=True)
    torch.save(out,OUT)
    print(f"  saved -> {OUT}  GPU peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB  total {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
