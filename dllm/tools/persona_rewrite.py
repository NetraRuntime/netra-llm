"""
Persona rewrite of assistant turns via Groq (openai/gpt-oss-120b): informal, warm,
chat-like voice — English and Indonesian system prompts, applied by source language.

Only prose-bearing sources are sent (smoltalk-en, hermes/* -> EN prompt; bactrian-id,
kolosal -> ID prompt). id-functioncall and xlam assistant turns are pure <tool_call>
JSON and are skipped entirely (the rewrite rules would return them unchanged).

Safety: every rewrite is verified — all <tool_call> blocks must survive byte-identically
(JSON-normalized compare) and the length ratio must stay sane; any violation falls back
to the original text. Identical texts are cached by hash (hermes is oversampled x2).
Progress is checkpointed to a JSONL cache so the job is resumable.

Usage:
    python dllm/tools/persona_rewrite.py \
        --messages_dir /data/datasets/sft-messages --output_dir /data/datasets/sft-messages-persona \
        --cache_path /data/datasets/raw/persona-cache.jsonl --concurrency 16
"""

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets

PROMPT_EN = """You rewrite text so it sounds natural, warm, and human, the way a real person talks in a chat, instead of stiff or corporate. You keep the meaning and structure exactly the same and only change the tone and phrasing.

WHAT TO REWRITE
- Natural-language text meant to be read by a person: assistant replies, messages, explanations, and reasoning written in prose.
- Inside tool or function calls, only rewrite argument values that are clearly human-facing messages (for example a "message", "reply", or "text" field that gets sent to an end user). Apply the same casual style to those.

WHAT TO KEEP EXACTLY (never change)
- All tool or function calls: function names, argument keys, and machine-facing values (IDs, enums, statuses, flags, channel names, query strings, parameters).
- Tool calls may appear in any format (JSON function calls, XML tags, or other notations). Keep that exact format.
- Code, JSON, URLs, file paths, email addresses, numbers, dates, prices, currency, math.
- Placeholders and variables: {name}, {{var}}, <slot>, %s, $VAR, and similar.
- Markdown structure, special tokens, and the overall format of the input.
- The language of the input. If a turn is in English, keep it English. Do not translate.

HARD RULES
- Do not add, remove, or change any information, fact, number, or step. Same content, different voice.
- Do not change what a tool call does or its arguments (except human-facing message text as noted above).
- Keep the same roles, turn order, and structure in multi-turn or agent data.
- Output only the rewritten content, in exactly the same format and structure as the input. No preamble, no explanation, no extra quotes or code fences that were not already in the input.
- Never use em dashes. Use commas, periods, parentheses, or colons instead.

STYLE (natural informal English)
- Use contractions: I'll, you're, can't, we've, let's, that's.
- Warm, direct openers and acknowledgments: Hey, Hi there, Sure, Got it, No worries, Sounds good.
- Short sentences, active voice, plain words.
- Real empathy instead of corporate apology: "Ah, sorry about that, let me fix it" instead of "We apologize for any inconvenience caused."
- Drop stiff phrases like: "please be advised", "kindly note", "at your earliest convenience", "thank you for reaching out to us", "we sincerely apologize".
- Stay polite and clear. Casual does not mean sloppy or unprofessional. No heavy slang unless the input already uses it. Add emoji only if the input already has them.

EXAMPLES

Input:
Thank you for contacting us. We sincerely apologize for any inconvenience this may have caused. Your refund request has been processed and will be reflected within 3-5 business days.
Output:
Thanks for reaching out, and sorry about the trouble! Your refund's all processed now. It should show up in 3 to 5 business days.

Input:
I will now check the status of your order.
[tool_call: get_order_status(order_id="ORD-48213")]
Your order has been shipped and is expected to arrive on March 14.
Output:
Let me check on your order real quick.
[tool_call: get_order_status(order_id="ORD-48213")]
Good news, it's already shipped. Should land around March 14.

Input:
[tool_call: send_reply(channel="whatsapp", message="We have received your inquiry and a representative will respond shortly.")]
Output:
[tool_call: send_reply(channel="whatsapp", message="Got your message! Someone will get back to you super soon.")]"""

PROMPT_ID = """Kamu menulis ulang teks supaya kedengeran natural, hangat, dan kayak manusia ngobrol di chat, bukan kaku atau korporat. Makna dan strukturnya dijaga sama persis, yang diubah cuma nada dan pemilihan katanya.

YANG DITULIS ULANG
- Teks bahasa natural yang dibaca manusia: balasan asisten, pesan, penjelasan, dan reasoning dalam bentuk prosa.
- Di dalam tool atau function call, cuma tulis ulang nilai argumen yang jelas-jelas pesan buat manusia (misalnya field "message", "reply", atau "text" yang dikirim ke user). Pakai gaya santai yang sama buat itu.

YANG DIJAGA SAMA PERSIS (jangan diubah)
- Semua tool atau function call: nama fungsi, key argumen, dan nilai yang dibaca mesin (ID, enum, status, flag, nama channel, query, parameter).
- Tool call bisa muncul dalam format apa pun (JSON function call, tag XML, atau notasi lain). Jaga formatnya persis.
- Kode, JSON, URL, path file, email, angka, tanggal, harga, mata uang, hitungan matematika.
- Placeholder dan variabel: {nama}, {{var}}, <slot>, %s, $VAR, dan sejenisnya.
- Struktur markdown, token khusus, dan format keseluruhan dari input.
- Bahasa asli input. Kalau ada bagian berbahasa Inggris (kode, istilah teknis), biarin. Jangan diterjemahin.

ATURAN WAJIB
- Jangan nambah, ngurangin, atau ngubah informasi, fakta, angka, atau langkah apa pun. Konten sama, gaya beda.
- Jangan ngubah fungsi tool call atau argumennya (kecuali teks pesan buat manusia seperti di atas).
- Jaga role, urutan turn, dan struktur yang sama di data multi-turn atau agent.
- Keluarkan cuma hasil tulis ulangnya, dengan format dan struktur sama persis kayak input. Tanpa pembuka, tanpa penjelasan, tanpa tambahan tanda kutip atau code fence yang nggak ada di input.
- Jangan pernah pakai em dash. Pakai koma, titik, tanda kurung, atau titik dua.

GAYA (bahasa Indonesia informal yang natural)
- Pakai bentuk santai: "nggak" atau "gak" (bukan "tidak"), "gimana" (bukan "bagaimana"), "aku" atau "kami", "makasih", "udah", "bentar".
- Kalau lagi nyapa atau ngomong langsung ke user, pakai sapaan ramah kayak "Kak". Tapi jangan dipaksa kalau konteksnya bukan nyapa orang.
- Kalimat pendek, langsung, kata-kata sederhana.
- Empati beneran, bukan minta maaf korporat: "wah, maaf banget ya soal ini, aku bantu beresin" bukan "kami mohon maaf atas ketidaknyamanan yang Anda alami".
- Buang frasa kaku: "mohon maaf atas ketidaknyamanannya", "dengan senang hati", "mohon menunggu", "terima kasih telah menghubungi kami".
- Partikel ("ya", "nih", "kok") buat ngehalusin, secukupnya, maksimal satu per kalimat.
- Tetap sopan dan jelas. Santai bukan berarti asal. Jangan slang berat (gue/lo, wkwk) kecuali inputnya emang udah gitu. Emoji cuma kalau inputnya udah ada.

CONTOH

Input:
Terima kasih telah menghubungi kami. Kami mohon maaf atas ketidaknyamanan yang Anda alami. Permintaan pengembalian dana Anda telah kami proses dan akan masuk dalam 3-5 hari kerja.
Output:
Makasih ya udah ngehubungin kami. Maaf banget soal kejadian ini. Refund-nya udah aku proses kok, masuknya sekitar 3 sampai 5 hari kerja ya.

Input:
Saya akan memeriksa status pesanan Anda terlebih dahulu.
[tool_call: get_order_status(order_id="ORD-48213")]
Pesanan Anda telah dikirim dan diperkirakan tiba pada 14 Maret.
Output:
Bentar ya, aku cek dulu status pesanannya.
[tool_call: get_order_status(order_id="ORD-48213")]
Udah dikirim kok! Perkiraan sampai sekitar 14 Maret ya.

Input:
[tool_call: send_reply(channel="whatsapp", message="Pesan Anda telah kami terima dan akan segera kami tanggapi.")]
Output:
[tool_call: send_reply(channel="whatsapp", message="Pesannya udah masuk ya, bentar lagi aku bales!")]"""

# which sources get rewritten, and with which prompt
SOURCE_LANG = {
    "smoltalk-en": "en",
    "hermes": "en",  # matches hermes/<config> via prefix
    "bactrian-id": "id",
    "kolosal": "id",
}

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
MODEL = "openai/gpt-oss-120b"


def _lang_for_source(source):
    for prefix, lang in SOURCE_LANG.items():
        if source == prefix or source.startswith(prefix + "/"):
            return lang
    return None


def _prose_len(text):
    """Length of the text once tool-call blocks are removed (is there prose to rewrite?)."""
    return len(_TOOL_CALL_RE.sub("", text).strip())


def _tool_calls_normalized(text):
    out = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            out.append(json.dumps(json.loads(m.group(1)), sort_keys=True))
        except json.JSONDecodeError:
            out.append(m.group(1).strip())
    return out


def _clean_dashes(text):
    """Replace em/en dashes with commas, but only OUTSIDE tool_call blocks."""
    parts, last = [], 0
    for m in _TOOL_CALL_RE.finditer(text):
        seg = text[last : m.start()]
        parts.append(re.sub(r"\s*[—–]\s*", ", ", seg))
        parts.append(text[m.start() : m.end()])
        last = m.end()
    parts.append(re.sub(r"\s*[—–]\s*", ", ", text[last:]))
    return "".join(parts)


def _verify(original, rewritten):
    """Accept the rewrite only if machine content survived and length is sane."""
    if not rewritten or not rewritten.strip():
        return False
    if _tool_calls_normalized(original) != _tool_calls_normalized(rewritten):
        return False
    ratio = len(rewritten) / max(1, len(original))
    if not (0.4 <= ratio <= 2.5):
        return False
    if "—" in rewritten or "–" in rewritten:  # em/en dash, banned
        return False
    return True


class Rewriter:
    def __init__(self, cache_path, concurrency, max_retries=6):
        from groq import Groq

        self.client = Groq()
        self.cache_path = cache_path
        self.cache = {}
        self.lock = threading.Lock()
        self.sem = threading.Semaphore(concurrency)
        self.max_retries = max_retries
        self.stats = {"api": 0, "cache": 0, "fallback": 0, "skipped": 0}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.cache[rec["k"]] = rec["v"]
                    except json.JSONDecodeError:
                        continue
            print(f"[persona] resumed cache: {len(self.cache):,} entries", flush=True)
        self.cache_f = open(cache_path, "a")

    def _call(self, prompt, text):
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                with self.sem:
                    resp = self.client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": text},
                        ],
                        temperature=1,
                        max_completion_tokens=8192,
                        top_p=1,
                        reasoning_effort="medium",
                        stream=False,
                    )
                return resp.choices[0].message.content or ""
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate" in msg.lower() or "503" in msg or "500" in msg:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                # any other API error (400/413/content policy/...): never kill the job —
                # log and fall back to the original text for this turn.
                print(f"[persona] WARN non-retryable: {type(e).__name__}: {msg[:140]}", flush=True)
                return ""
        return ""  # give up -> fallback to original

    def rewrite(self, text, lang):
        if _prose_len(text) < 16:
            self.stats["skipped"] += 1
            return text
        key = hashlib.sha1((lang + "\x00" + text).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                self.stats["cache"] += 1
                return self.cache[key] if self.cache[key] is not None else text
        out = self._call(PROMPT_EN if lang == "en" else PROMPT_ID, text)
        out = _clean_dashes(out)
        ok = _verify(text, out)
        value = out if ok else None  # None = verified-failed -> original (cached too)
        with self.lock:
            self.cache[key] = value
            self.cache_f.write(json.dumps({"k": key, "v": value}) + "\n")
            self.cache_f.flush()
            self.stats["api"] += 1
            if not ok:
                self.stats["fallback"] += 1
        return out if ok else text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--messages_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_path", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="rewrite only first N rows (debug)")
    args = p.parse_args()

    ds = datasets.load_from_disk(args.messages_dir)
    rw = Rewriter(args.cache_path, args.concurrency)

    def process_row(row):
        lang = _lang_for_source(row.get("source", ""))
        if lang is None:
            return row
        msgs = [dict(m) for m in row["messages"]]
        for m in msgs:
            if m["role"] == "assistant":
                m["content"] = rw.rewrite(m["content"], lang)
        return {"messages": msgs, "source": row["source"]}

    out = {}
    for split, d in ds.items():
        rows = list(d)
        if args.limit:
            rows = rows[: args.limit]
        results = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(process_row, r): i for i, r in enumerate(rows)}
            done = 0
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
                done += 1
                if done % 500 == 0:
                    print(f"[persona] {split}: {done:,}/{len(rows):,}  stats={rw.stats}", flush=True)
        out[split] = datasets.Dataset.from_list(results)
        print(f"[persona] {split} DONE: {len(results):,} rows  stats={rw.stats}", flush=True)

    datasets.DatasetDict(out).save_to_disk(args.output_dir)
    print(f"[persona] saved -> {args.output_dir}  final stats={rw.stats}", flush=True)


if __name__ == "__main__":
    main()
