# P1 stage-1 (RESEARCH): encode src/edit prompts with T5-XXL, save embeds, EXIT.
# OOM-safe: only the T5 ENCODER (~3.7B -> ~8GB bf16) is materialized; safetensors mmap;
# run inside a MemoryMax-capped user scope so the fp32 page-cache read is reclaimed, not
# leaked to the session. Frees everything on process exit before the DiT stage loads.
import os, time, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from transformers import T5EncoderModel, AutoTokenizer

REPO="google/t5-v1_1-xxl"; DEV,DT="cuda",torch.bfloat16
MAXLEN=128
SRC ="a red sports car driving on a coastal road"
EDIT="a blue sports car driving on a coastal road"
OUT ="/home/alex/swiftedit-ltx-research/p1_embeds.pt"

def main():
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(REPO)
    print("loading T5 encoder (bf16, encoder-only)...",flush=True)
    enc=T5EncoderModel.from_pretrained(REPO, torch_dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    print(f"  loaded in {time.time()-t0:.0f}s  GPU={torch.cuda.memory_allocated()/1e9:.1f}GB",flush=True)
    def embed(p):
        b=tok([p], max_length=MAXLEN, padding="max_length", truncation=True, return_tensors="pt")
        ids=b.input_ids.to(DEV); m=b.attention_mask.to(DEV)
        with torch.no_grad(): h=enc(ids, attention_mask=m).last_hidden_state
        return h.cpu(), m.cpu()
    hs,ms = embed(SRC); he,me = embed(EDIT)
    torch.save({"src":hs,"src_mask":ms,"edit":he,"edit_mask":me,
                "src_prompt":SRC,"edit_prompt":EDIT,"maxlen":MAXLEN}, OUT)
    print(f"  src embeds {tuple(hs.shape)}  edit embeds {tuple(he.shape)}",flush=True)
    print(f"  saved -> {OUT}",flush=True)
    print(f"  GPU peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB  total {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
