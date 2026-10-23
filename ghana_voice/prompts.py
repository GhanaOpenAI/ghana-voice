"""
Canonical language-specific test prompts for synthesizing diverse, natural speech.
Includes curated conversational phrases for major languages and an automatic
held-out evaluation extractor for all 45+ Ghanaian languages.
"""

from pathlib import Path

LANGUAGE_SAMPLE_PROMPTS = {
    "asante_twi": [
        ("01_greeting", "Wo ho te sɛn? Me ho yɛ, na wo nso ɛ?"),
        ("02_welcome", "Akwaaba, yɛbɔ mpae sɛ Onyankopɔn bɛhyira wo ne wo fiefoɔ nyinaa."),
        ("03_school", "Me kɔ sukuu da biara na mepɛ sɛ mesua ade pa ara."),
        ("04_proverb", "Woforo dua pa a, na yɛpia wo."),
        ("05_story", "Kofi ne Ama kɔɔ fie kɔbɔɔ amanneɛ kyerɛɛ wɔn awofoɔ nyinaa."),
        ("06_request", "Yɛsrɛ wo, bue pono no ma me kakra na mframa mmra mu."),
    ],
    "akuapem_twi": [
        ("01_greeting", "Wo ho te dɛn? Me ho yɛ pɛpɛɛpɛ, na wo nso ɛ?"),
        ("02_morning", "Maakye, me ma wo awurade nhyira da kɛse yi mu."),
        ("03_school", "Me kɔ sukuu da biara na mepɛ sɛ mesua ade yiye."),
        ("04_scripture", "Onyankopɔn yɛ ɔdɔ na ɔdɔ obiara wɔ wiase nyinaa."),
        ("05_proverb", "Baanu so a, ɛmmoa na ɛso yɛ hare."),
        ("06_request", "Mepa wo kyɛw, fa nsuo no ma me na mennom kakra."),
    ],
    "fante": [
        ("01_greeting", "Me ma wo akye, wo ho te dɛn ndɛ?"),
        ("02_praise", "Nyame yɛ kɛse na ɔwɔ tumi nyinaa wɔ sor ne fam."),
        ("03_school", "Mbofra no kɔ skuul dabiara dɛ wɔrokɔsua adze pa."),
        ("04_proverb", "Nyansa nyɛ sika na wɔakyekyere asie."),
        ("05_dialogue", "Ebue pon no ma me kakra ama mframa ahyɛ dan no mu."),
        ("06_family", "Yɛbɔ mbɔden dabiara ama hɛn mbusuakuw anya ahotɔ."),
    ],
    "ga": [
        ("01_greeting", "Te oyoo tɛŋŋ? Miye jogbaŋŋ, oyiwaladɔŋŋ."),
        ("02_praise", "Nyɔŋmɔ ji mi nɔkwɛlɔ, nɔ ko nɔ ko ehiaŋ mi."),
        ("03_day", "Gbi nɛɛ ji gbi kpakpa ni Nyɔŋmɔ efee kɛha wɔ."),
        ("04_school", "Miyaa skul daa gbi koni mayakase nii babaoo."),
        ("05_gratitude", "Wɔmii shidaa babaoo kɛha nyɛ mɔbɔnalɛ kɛ suɔmɔ lɛ."),
        ("06_dialogue", "Baa biɛ nɛɛ ni okɛ mi awie he ko he ko."),
    ],
    "ewe": [
        ("01_greeting", "Ndi na wò, èfɔ nyuiea? Mefɔ nyuie, akpe kaka."),
        ("02_shepherd", "Yehowa nye nye kplɔla, naneke mehiãam o."),
        ("03_school", "Deviwo yina sukuu gbe sia gbe be woasrɔ̃ nu nyuie."),
        ("04_love", "Mawu nye lɔlɔ̃ eye wòlɔ̃ amesiame le xexeame godoo."),
        ("05_peace", "Miawo katã míele agbagba dzem be míakpɔ dzidzedze kple ŋutifafa."),
        ("06_dialogue", "Va afisia ne míaƒo nu le míaƒe dɔwɔwɔ ŋu."),
    ],
    "dagbani": [
        ("01_greeting", "Dasiba, a biem be wula? N biem nyɛla viɛnyɛla."),
        ("02_shepherd", "Naawuni n-nyɛ n gubi, shɛli bi kpaŋsi ma."),
        ("03_school", "Bia maa chana shikuru dabsili kam ni o ti bo baŋsim."),
        ("04_grace", "Naawuni nambɔzɔbo galisiya pam n-ti ti zaa."),
        ("05_nation", "Ti bɔrimi ni ti tum tuun viɛla ti tiŋgbani ŋɔ puuni."),
        ("06_dialogue", "Kpama dooma kamani ka ti laɣim ka nina."),
    ],
    "dagaare": [
        ("01_greeting", "Fo bebie e la soŋ? N bebie e la soŋ yaga."),
        ("02_shepherd", "Naabmenɛ la n gɔɔro, boma zaa ba pɔgɔ ma."),
        ("03_school", "Bebie zaa n gaa la sakuuri ka n te zanne baŋkpɛ."),
        ("04_blessing", "Te Naabmenɛ na tɔɔrɔ la kpeɛoŋ ko te zaa."),
        ("05_dialogue", "Wa ŋa ka te yele a yele ŋa yaga."),
    ],
    "frafra": [
        ("01_greeting", "Bulika, ya kɔma ani seŋ? Ti kɔma ani sonla."),
        ("02_shepherd", "Naayine de la n gɔɔra, ba buse bɔna kaŋa."),
        ("03_school", "Koma la le koli sakuure dabserɛ woo ti zamesɛ gɔŋɔ."),
        ("04_praise", "Naayine tɔm de la zuma n boi ti zaa bɔŋɔ."),
    ],
    "nzema": [
        ("01_greeting", "Kɛzi ɛte ɛ? Me nwo yɛ, mo kpalɛ yaga."),
        ("02_shepherd", "Nyamenle a le me neavolɛ, ɛhwee ɛngyia me."),
        ("03_school", "Kɔ sukulu dahuu na sukua nrɛlɛbɛ kpalɛ."),
        ("04_blessing", "Nyamenle hyira yɛ muala wɔ mekɛ kpalɛ nwo."),
    ],
    "ahanta": [
        ("01_greeting", "Mɔkye, wo nwo te sɛn? Me nwo yɛ pɛpɛɛpɛ."),
        ("02_shepherd", "Nyamenle a le me neavolɛ, debie bia ɛngyia me."),
        ("03_school", "Mbofra nɔ kɔ sukulu dahuu dɛ bɛkɔsukua ndzɛmba kpalɛ."),
    ],
    "sehwi": [
        ("01_greeting", "Wo nwo te sɛn? Me nwo le kpalɛ pa ara."),
        ("02_shepherd", "Nyame le me neanvolɛ, hwee ɛngyia me da."),
        ("03_school", "Mbofra kɔ sukulu dahuu sɛ bɛkɔzua debie kpalɛ."),
    ],
    "dangme": [
        ("01_greeting", "Kɛ o ngɛ kɛɛ? I ngɛ saminya, oyiwaladɔŋŋ."),
        ("02_shepherd", "Mawu ji ye nɔkwɛlɔ kpakpa, nɔ ko be ni maa hia mi."),
        ("03_school", "Bimɛ ɔmɛ yaa sukuu daa ligbi kɛkase níhi babauu."),
    ],
    "ghana_english": [
        ("01_welcome", "Welcome to Ghana, the gateway to Africa and the home of peace."),
        ("02_morning", "Good morning, how are you doing today? I hope you have a productive day."),
        ("03_education", "Education is the foundation for unlocking the true potential of our continent."),
        ("04_unity", "We must work together to build a prosperous and united nation."),
        ("05_dialogue", "Could you please tell me more about the project you are working on?"),
        ("06_culture", "Ghanaian hospitality and warmth are known and celebrated all around the world."),
    ],
}


def get_prompts_for_language(lang_name: str, dataset_dir: str = None):
    """
    Retrieve language-specific prompts.
    First checks curated conversational phrases.
    If not found, dynamically pulls held-out test sentences directly from the language's clean dataset.
    """
    cleaned = lang_name.lower().strip()
    for k, v in LANGUAGE_SAMPLE_PROMPTS.items():
        if k in cleaned or cleaned in k:
            return v

    # Fallback to dynamic held-out sentences from dataset
    candidate_paths = [
        Path(f"/mnt/volume_d2wey28/projects/vieneu-ghana/dataset_10h/{cleaned}/metadata_encoded.csv"),
        Path(f"/mnt/volume_d2wey28/projects/vieneu-ghana/dataset_languages/{cleaned}/metadata_encoded.csv"),
    ]
    if dataset_dir:
        candidate_paths.insert(0, Path(dataset_dir) / "metadata_encoded.csv")

    for p in candidate_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                # Grab 6 diverse sentences from the validation tail
                extracted = []
                tail = lines[-30:]  # last 30 lines
                step = max(1, len(tail) // 6)
                for i in range(0, min(len(tail), 6 * step), step):
                    parts = tail[i].split("|")
                    if len(parts) >= 2:
                        ipa_text = parts[1].strip()
                        extracted.append((f"0{len(extracted)+1}_eval", ipa_text))
                    if len(extracted) >= 6:
                        break
                if extracted:
                    return extracted
            except Exception:
                pass

    return LANGUAGE_SAMPLE_PROMPTS["asante_twi"]


def generate_samples_for_language(
    model_dir: str,
    lang_name: str,
    out_dir: str,
    gguf_path: str = None,
    device: str = "cuda:0",
    count: int = 3,
):
    """
    Synthesize authentic evaluation audio samples in BOTH formats:
    1. PyTorch (GPU / CUDA)
    2. GGUF Q8_0 (CPU / llama.cpp)
    """
    import os
    from pathlib import Path
    import soundfile as sf
    from .engine import GhanaVoice
    from .constants import SAMPLE_RATE_24K

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prompts = get_prompts_for_language(lang_name)
    n_samples = min(count, len(prompts))
    print(f"[Sample Generator] Synthesizing {n_samples} benchmark samples from {model_dir} for {lang_name}...")

    # 1. PyTorch GPU Synthesis
    try:
        print("[Sample Generator] Loading PyTorch model on GPU ...")
        engine_pt = GhanaVoice(lang=lang_name, model=model_dir, device=device)
        for tag, text in prompts[:count]:
            print(f"  PyTorch GPU [{tag}]: {text}")
            wav_pt = engine_pt.synthesize(text, temperature=0.7, top_p=0.8, top_k=50, repetition_penalty=1.0)
            sf.write(str(out_path / f"sample_{tag}_pytorch.wav"), wav_pt, SAMPLE_RATE_24K)
    except Exception as e:
        print(f"  Warning during PyTorch synthesis: {e}")

    # 2. GGUF Q8_0 CPU Synthesis
    # Locate GGUF Q8_0 file if not explicitly given
    if not gguf_path:
        m_path = Path(model_dir)
        candidates = [
            m_path.parent / "gguf" / f"ghana-voice_{lang_name}-Q8_0.gguf",
            m_path.parent / "gguf" / f"{m_path.name}-Q8_0.gguf",
            m_path / f"ghana-voice_{lang_name}-Q8_0.gguf",
        ]
        for c in candidates:
            if c.exists():
                gguf_path = str(c)
                break

    if gguf_path and os.path.exists(gguf_path):
        try:
            print(f"[Sample Generator] Loading GGUF Q8_0 model on CPU: {gguf_path} ...")
            engine_gg = GhanaVoice(lang=lang_name, model=gguf_path, device="cpu", gguf_type="Q8_0")
            for tag, text in prompts[:count]:
                print(f"  GGUF Q8_0 CPU [{tag}]: {text}")
                wav_gg = engine_gg.synthesize(text, temperature=0.7, top_p=0.8, top_k=50, repetition_penalty=1.0)
                sf.write(str(out_path / f"sample_{tag}_gguf_q8.wav"), wav_gg, SAMPLE_RATE_24K)
        except Exception as e:
            print(f"  Warning during GGUF synthesis: {e}")

    # Also save standard sample_01.wav aliases pointing to pytorch
    for tag, _ in prompts[:count]:
        pt_f = out_path / f"sample_{tag}_pytorch.wav"
        st_f = out_path / f"sample_{tag}.wav"
        if pt_f.exists() and not st_f.exists():
            import shutil
            shutil.copy(str(pt_f), str(st_f))

