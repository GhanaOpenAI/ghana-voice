"""
Canonical language-specific test prompts for synthesizing diverse, natural speech.
Every language has multiple native sentences (greetings, everyday dialogue, questions, proverbs).
"""

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


def get_prompts_for_language(lang_name: str):
    """Retrieve language-specific prompts, falling back to Twi if unknown."""
    cleaned = lang_name.lower().strip()
    for k, v in LANGUAGE_SAMPLE_PROMPTS.items():
        if k in cleaned or cleaned in k:
            return v
    return LANGUAGE_SAMPLE_PROMPTS["asante_twi"]
