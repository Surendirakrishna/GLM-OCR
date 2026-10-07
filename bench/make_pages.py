"""Build a small OCR benchmark: English + Indic text and tables, with ground truth.

Each page is rendered as HTML in Chromium (correct Indic shaping) at ~200 DPI
A4, then saved twice: ``<name>.png`` (clean) and ``<name>_scan.png``
(grayscale, slight skew, blur, noise, JPEG artefacts). Ground truth comes from
the same data that renders the page, in ``gt/<name>.json``:

    {"text": [str, ...],   # headings/paragraphs/list items in reading order
     "tables": [html, ...]} # each table as HTML (rowspan/colspan kept)

Usage: python bench/make_pages.py [OUT_DIR]   (default bench/data)
"""

from __future__ import annotations

import html
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageFilter

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("data"))
PAGE_W = 1654  # A4 at 200 DPI
PAGE_H = 2339

FONTS = {
    "en": "'Noto Sans', 'DejaVu Sans', sans-serif",
    "hi": "'Noto Sans Devanagari', 'Noto Sans', sans-serif",
    "ta": "'Noto Sans Tamil', 'Noto Sans', sans-serif",
    "bn": "'Noto Sans Bengali', 'Noto Sans', sans-serif",
    "te": "'Noto Sans Telugu', 'Noto Sans', sans-serif",
    "kn": "'Noto Sans Kannada', 'Noto Sans', sans-serif",
    "ml": "'Noto Sans Malayalam', 'Noto Sans', sans-serif",
    "gu": "'Noto Sans Gujarati', 'Noto Sans', sans-serif",
}

rng = random.Random(7)


def inr(value: float, decimals: int = 2) -> str:
    """Indian digit grouping: 12,34,567.89; negatives in parentheses."""
    neg = value < 0
    whole, _, frac = f"{abs(value):.{decimals}f}".partition(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    s = ",".join(groups + [tail]) if groups else tail
    if frac:
        s += "." + frac
    return f"({s})" if neg else s


BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


# ----------------------------------------------------------------------
# Page builders: each returns (body_html, gt)
# ----------------------------------------------------------------------


class Page:
    def __init__(self):
        self.parts: list[str] = []
        self.text: list[str] = []
        self.tables: list[str] = []

    def h(self, text, level=2):
        self.parts.append(f"<h{level}>{html.escape(text)}</h{level}>")
        self.text.append(text)

    def p(self, text, cls=""):
        self.parts.append(f'<p class="{cls}">{html.escape(text)}</p>')
        self.text.append(text)

    def ul(self, items):
        self.parts.append("<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>")
        self.text.extend(items)

    def table(self, rows, cls="grid"):
        """rows: list of rows; a cell is str or (str, {"rowspan":..,"colspan":..,"th":bool,"b":bool})."""
        out = []
        for row in rows:
            cells = []
            for cell in row:
                text, opts = (cell, {}) if isinstance(cell, str) else cell
                tag = "th" if opts.get("th") else "td"
                attrs = "".join(
                    f' {k}="{opts[k]}"' for k in ("rowspan", "colspan") if opts.get(k, 1) != 1
                )
                body = html.escape(text)
                if opts.get("b"):
                    body = f"<b>{body}</b>"
                cells.append(f"<{tag}{attrs}>{body}</{tag}>")
            out.append("<tr>" + "".join(cells) + "</tr>")
        t = "<table>" + "".join(out) + "</table>"
        self.parts.append(f'<table class="{cls}">' + "".join(out) + "</table>")
        self.tables.append(t)

    def raw(self, markup):
        self.parts.append(markup)

    def gt(self):
        return {"text": self.text, "tables": self.tables}


def th(t, **kw):
    return (t, {"th": True, **kw})


def en_financial():
    pg = Page()
    pg.h("Statement of Profit and Loss for the year ended 31 March 2024", 1)
    pg.p(
        "The Company recorded its highest ever revenue from operations during the year, "
        "supported by higher throughput at both refineries and improved marketing margins. "
        "All figures are in ₹ crore unless otherwise stated."
    )
    rows = [
        [th("Particulars", rowspan=2), th("Note", rowspan=2), th("Standalone", colspan=2), th("Consolidated", colspan=2)],
        [th("FY 2023-24"), th("FY 2022-23"), th("FY 2023-24"), th("FY 2022-23")],
    ]
    items = [
        ("Revenue from Operations", "24"), ("Other Income", "25"), ("Total Income", ""),
        ("Cost of materials consumed", "26"), ("Purchase of stock-in-trade", "27"),
        ("Employee benefits expense", "28"), ("Finance costs", "29"),
        ("Depreciation and amortisation expense", "30"), ("Other expenses", "31"),
        ("Profit before exceptional items and tax", ""), ("Exceptional items", "32"),
        ("Tax expense", "33"), ("Profit for the year", ""),
    ]
    for name, note in items:
        bold = name.startswith(("Total", "Profit for"))
        vals = []
        for _ in range(4):
            v = rng.uniform(800, 480000)
            if name == "Exceptional items":
                v = -rng.uniform(100, 2000)
            vals.append(inr(v))
        rows.append([(name, {"b": bold}), note] + [(v, {"b": bold}) for v in vals])
    pg.table(rows)
    pg.p("Note: Figures in brackets denote negative values. Previous year figures have been regrouped wherever necessary.", "small")
    return pg


def en_dense():
    pg = Page()
    pg.h("Annexure III: State-wise release and utilisation of funds (₹ lakh)", 2)
    states = {
        "Southern": ["Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana"],
        "Western": ["Goa", "Gujarat", "Maharashtra", "Rajasthan"],
        "Eastern": ["Bihar", "Jharkhand", "Odisha", "West Bengal"],
    }
    rows = [[th("Region"), th("Sl."), th("State"), th("Opening Balance"), th("Released"),
             th("Utilised"), th("Closing Balance"), th("Utilisation %")]]
    sl = 1
    for region, names in states.items():
        for i, name in enumerate(names):
            ob, rel = rng.uniform(100, 9000), rng.uniform(1000, 60000)
            used = rel * rng.uniform(0.55, 0.99)
            cb = ob + rel - used
            cells = [str(sl), name, inr(ob), inr(rel), inr(used), inr(cb), f"{used / (ob + rel) * 100:.1f}"]
            row = ([(region, {"rowspan": len(names)})] if i == 0 else []) + cells
            rows.append(row)
            sl += 1
    pg.table(rows, "grid dense")
    pg.p("Source: Utilisation certificates received up to 30 June 2024.", "small")
    return pg


def en_borderless():
    pg = Page()
    pg.h("Key Performance Indicators", 2)
    pg.p("The table below summarises operating performance for the last three financial years.")
    rows = [[th("Indicator"), th("Unit"), th("2021-22"), th("2022-23"), th("2023-24")]]
    for ind, unit in [("Crude throughput", "MMT"), ("Distillate yield", "%"), ("Gross refining margin", "USD/bbl"),
                      ("Market sales", "MMT"), ("Retail outlets", "Nos."), ("Pipeline throughput", "MMT"),
                      ("Specific energy consumption", "MBN"), ("Employees", "Nos.")]:
        if unit == "Nos.":
            vals = [inr(rng.randint(8000, 22000), 0) for _ in range(3)]
        else:
            vals = [f"{rng.uniform(2, 95):.2f}" for _ in range(3)]
        rows.append([ind, unit] + vals)
    pg.table(rows, "borderless")
    pg.h("Highlights", 3)
    pg.ul([
        "Commissioned the 9 MMTPA refinery expansion project ahead of schedule.",
        "Added 1,152 new retail outlets, taking the network to 21,031 outlets.",
        "Reduced specific energy consumption by 3.4% over the previous year.",
    ])
    return pg


HI_PARAS = [
    "भारत सरकार ने वित्तीय वर्ष 2023-24 के लिए ग्रामीण विकास योजनाओं के अंतर्गत कुल ₹1,24,500 करोड़ का बजट आवंटित किया है। इस राशि का उपयोग सड़क निर्माण, पेयजल आपूर्ति और स्वास्थ्य केंद्रों के विस्तार के लिए किया जाएगा।",
    "लेखापरीक्षा के दौरान यह पाया गया कि कई जिलों में स्वीकृत परियोजनाएँ निर्धारित समय सीमा के भीतर पूरी नहीं हुईं। विभाग को निर्देश दिया गया है कि वह शेष कार्यों की प्रगति रिपोर्ट तीन महीने के भीतर प्रस्तुत करे।",
    "राज्य सरकारों से अनुरोध है कि वे उपयोगिता प्रमाणपत्र समय पर भेजें ताकि अगली किस्त जारी की जा सके। किसी भी प्रकार की देरी की स्थिति में संबंधित अधिकारी की जवाबदेही तय की जाएगी।",
]


def hi_en_bilingual():
    pg = Page()
    pg.h("ग्रामीण विकास मंत्रालय / Ministry of Rural Development", 1)
    pg.p(HI_PARAS[0])
    pg.p("The Ministry has allocated a total budget of ₹1,24,500 crore for rural development schemes for the financial year 2023-24.")
    rows = [[th("क्रम सं.<br>Sl. No.".replace("<br>", " / ")), th("योजना का नाम / Scheme"), th("स्वीकृत राशि (₹ करोड़)"),
             th("व्यय (₹ करोड़)"), th("प्रतिशत / %")]]
    ta = te = 0.0
    for i, s in enumerate(["प्रधानमंत्री ग्राम सड़क योजना (PMGSY)", "जल जीवन मिशन (JJM)", "राष्ट्रीय स्वास्थ्य मिशन (NHM)",
                           "महात्मा गांधी नरेगा (MGNREGA)", "प्रधानमंत्री आवास योजना - ग्रामीण"], 1):
        a = round(rng.uniform(5000, 90000), 2)
        e = round(a * rng.uniform(0.6, 0.98), 2)
        ta, te = ta + a, te + e
        rows.append([str(i), s, inr(a), inr(e), f"{e / a * 100:.2f}"])
    rows.append([("कुल / Total", {"colspan": 2, "b": True}), (inr(ta), {"b": True}),
                 (inr(te), {"b": True}), (f"{te / ta * 100:.2f}", {"b": True})])
    pg.table(rows)
    pg.p(HI_PARAS[1])
    return pg


def hi_two_column():
    pg = Page()
    pg.h("लेखापरीक्षा प्रतिवेदन: मुख्य निष्कर्ष", 1)
    left = [HI_PARAS[1], HI_PARAS[2]]
    right = [
        "निरीक्षण में यह भी देखा गया कि पंचायत स्तर पर अभिलेखों का रखरखाव ठीक प्रकार से नहीं किया जा रहा था। कई मामलों में भुगतान बिना माप पुस्तिका के सत्यापन के किया गया।",
        "समिति ने सिफारिश की है कि सभी भुगतानों को सार्वजनिक वित्तीय प्रबंधन प्रणाली के माध्यम से किया जाए और प्रत्येक तिमाही में सामाजिक लेखापरीक्षा आयोजित की जाए।",
    ]
    col = lambda ps: "".join(f"<p>{html.escape(p)}</p>" for p in ps)
    pg.raw(f'<div class="cols"><div>{col(left)}</div><div>{col(right)}</div></div>')
    pg.text.extend(left + right)
    return pg


INDIC = {
    "ta": dict(
        title="பொதுப்பணித் துறை – நிதி ஒதுக்கீடு 2023-24",
        paras=[
            "தமிழ்நாடு அரசு 2023-24 ஆம் நிதியாண்டில் பொதுப்பணித் துறைக்கு ₹18,250 கோடி நிதி ஒதுக்கீடு செய்துள்ளது. இந்த நிதி சாலைகள், பாலங்கள் மற்றும் அரசு கட்டடங்களின் பராமரிப்புக்குப் பயன்படுத்தப்படும்.",
            "தணிக்கையின்போது சில மாவட்டங்களில் ஒப்பந்தப் பணிகள் குறித்த காலத்திற்குள் முடிக்கப்படவில்லை என்பது கண்டறியப்பட்டது.",
        ],
        head=["வ.எண்", "மாவட்டம்", "ஒதுக்கீடு (₹ கோடி)", "செலவு (₹ கோடி)"],
        rows=["சென்னை", "கோயம்புத்தூர்", "மதுரை", "திருச்சிராப்பள்ளி", "சேலம்", "திருநெல்வேலி"],
    ),
    "bn": dict(
        title="শিক্ষা দপ্তর – বাজেট বরাদ্দ ২০২৩-২৪",
        paras=[
            "পশ্চিমবঙ্গ সরকার ২০২৩-২৪ অর্থবর্ষে শিক্ষা খাতে মোট ৩২,৪০০ কোটি টাকা বরাদ্দ করেছে। এই অর্থ বিদ্যালয় ভবন নির্মাণ, শিক্ষক নিয়োগ এবং মিড-ডে মিল প্রকল্পে ব্যয় করা হবে।",
            "নিরীক্ষায় দেখা গেছে যে কয়েকটি জেলায় বরাদ্দকৃত অর্থ সময়মতো ব্যবহার করা হয়নি।",
        ],
        head=["ক্রমিক নং", "জেলা", "বরাদ্দ (কোটি টাকা)", "ব্যয় (কোটি টাকা)"],
        rows=["কলকাতা", "হাওড়া", "দার্জিলিং", "মুর্শিদাবাদ", "বর্ধমান"],
        native_digits=True,
    ),
    "te": dict(
        title="వ్యవసాయ శాఖ – నిధుల కేటాయింపు",
        paras=[
            "ఆంధ్రప్రదేశ్ ప్రభుత్వం 2023-24 ఆర్థిక సంవత్సరంలో వ్యవసాయ రంగానికి ₹41,436 కోట్లు కేటాయించింది. ఈ నిధులను నీటిపారుదల ప్రాజెక్టులు, విత్తనాల సరఫరా మరియు రైతు భరోసా పథకానికి వినియోగిస్తారు.",
        ],
        head=["క్ర.సం.", "జిల్లా", "కేటాయింపు (₹ కోట్లు)", "వ్యయం (₹ కోట్లు)"],
        rows=["విశాఖపట్నం", "గుంటూరు", "కృష్ణా", "నెల్లూరు", "కర్నూలు"],
    ),
    "kn": dict(
        title="ಆರೋಗ್ಯ ಇಲಾಖೆ – ಅನುದಾನ ವಿವರ",
        paras=[
            "ಕರ್ನಾಟಕ ಸರ್ಕಾರವು 2023-24ನೇ ಹಣಕಾಸು ವರ್ಷದಲ್ಲಿ ಆರೋಗ್ಯ ಇಲಾಖೆಗೆ ₹14,950 ಕೋಟಿ ಅನುದಾನ ನೀಡಿದೆ. ಈ ಹಣವನ್ನು ಪ್ರಾಥಮಿಕ ಆರೋಗ್ಯ ಕೇಂದ್ರಗಳ ಉನ್ನತೀಕರಣ ಮತ್ತು ಔಷಧಿ ಖರೀದಿಗೆ ಬಳಸಲಾಗುವುದು.",
        ],
        head=["ಕ್ರ.ಸಂ.", "ಜಿಲ್ಲೆ", "ಅನುದಾನ (₹ ಕೋಟಿ)", "ವೆಚ್ಚ (₹ ಕೋಟಿ)"],
        rows=["ಬೆಂಗಳೂರು", "ಮೈಸೂರು", "ಬೆಳಗಾವಿ", "ಕಲಬುರಗಿ", "ಮಂಗಳೂರು"],
    ),
}


def indic_page(lang):
    spec = INDIC[lang]
    pg = Page()
    pg.h(spec["title"], 1)
    pg.p(spec["paras"][0])
    rows = [[th(h) for h in spec["head"]]]
    for i, d in enumerate(spec["rows"], 1):
        a = rng.uniform(200, 6000)
        e = a * rng.uniform(0.5, 0.97)
        cells = [str(i), d, inr(a), inr(e)]
        if spec.get("native_digits"):
            cells = [c.translate(BN_DIGITS) if j != 1 else c for j, c in enumerate(cells)]
        rows.append(cells)
    pg.table(rows)
    for para in spec["paras"][1:]:
        pg.p(para)
    return pg


def ml_gu_text():
    pg = Page()
    pg.h("സംസ്ഥാന ബജറ്റ് / રાજ્ય બજેટ", 1)
    pg.p("കേരള സർക്കാർ 2023-24 സാമ്പത്തിക വർഷത്തിൽ ആരോഗ്യ മേഖലയ്ക്ക് ₹10,200 കോടി രൂപ അനുവദിച്ചു. ഈ തുക ആശുപത്രികളുടെ നവീകരണത്തിനും മരുന്നുകൾ വാങ്ങുന്നതിനും ഉപയോഗിക്കും.")
    pg.p("ഓഡിറ്റിൽ ചില ജില്ലകളിൽ അനുവദിച്ച തുക സമയബന്ധിതമായി ചെലവഴിച്ചിട്ടില്ലെന്ന് കണ്ടെത്തി.")
    pg.h("ગુજરાત – શિક્ષણ વિભાગ", 2)
    pg.p("ગુજરાત સરકારે વર્ષ 2023-24 માટે શિક્ષણ વિભાગને ₹43,651 કરોડની ફાળવણી કરી છે. આ રકમનો ઉપયોગ શાળાઓના માળખાકીય વિકાસ અને શિક્ષકોની તાલીમ માટે કરવામાં આવશે.")
    pg.ul([
        "પ્રાથમિક શાળાઓમાં 12,000 નવા વર્ગખંડોનું નિર્માણ.",
        "35,000 શિક્ષકો માટે ડિજિટલ તાલીમ કાર્યક્રમ.",
    ])
    return pg


PAGES = {
    "en_financial": (en_financial, "en"),
    "en_dense": (en_dense, "en"),
    "en_borderless": (en_borderless, "en"),
    "hi_en_bilingual": (hi_en_bilingual, "hi"),
    "hi_two_column": (hi_two_column, "hi"),
    "ta_table": (lambda: indic_page("ta"), "ta"),
    "bn_table": (lambda: indic_page("bn"), "bn"),
    "te_table": (lambda: indic_page("te"), "te"),
    "kn_table": (lambda: indic_page("kn"), "kn"),
    "ml_gu_text": (ml_gu_text, "ml"),
}

CSS = """
body { margin: 0; background: white; }
.page { width: %(w)dpx; min-height: %(h)dpx; box-sizing: border-box; padding: 150px 160px;
        font-family: %(font)s; font-size: 27px; line-height: 1.5; color: #111; }
h1 { font-size: 40px; margin: 0 0 30px; } h2 { font-size: 34px; margin: 30px 0 20px; }
h3 { font-size: 30px; } p { margin: 0 0 24px; text-align: justify; } .small { font-size: 22px; }
table { border-collapse: collapse; width: 100%%; margin: 10px 0 30px; font-size: 24px; }
table.grid th, table.grid td { border: 2px solid #333; padding: 8px 12px; }
table.dense { font-size: 20px; } table.dense th, table.dense td { padding: 4px 8px; }
table.borderless th { border-bottom: 2px solid #333; text-align: left; padding: 8px 12px; }
table.borderless td { padding: 8px 12px; }
th { background: #e8eef6; } td:not(:nth-child(-n+2)) { text-align: right; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 60px; }
"""


def degrade(img: Image.Image, seed: int) -> Image.Image:
    r = random.Random(seed)
    img = img.convert("L").rotate(r.uniform(-0.8, 0.8), resample=Image.BICUBIC, expand=False, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(0.9))
    noise = Image.effect_noise(img.size, 18)
    img = Image.blend(img, noise, 0.08)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=55)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def main():
    from playwright.sync_api import sync_playwright

    (OUT / "gt").mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": PAGE_W, "height": PAGE_H})
        for i, (name, (build, lang)) in enumerate(PAGES.items()):
            pg = build()
            doc = (
                "<html><head><meta charset='utf-8'><style>"
                + CSS % {"w": PAGE_W, "h": PAGE_H, "font": FONTS[lang]}
                + "</style></head><body><div class='page'>"
                + "".join(pg.parts)
                + "</div></body></html>"
            )
            page.set_content(doc, wait_until="networkidle")
            png = page.screenshot(clip={"x": 0, "y": 0, "width": PAGE_W, "height": PAGE_H})
            img = Image.open(io.BytesIO(png)).convert("RGB")
            img.save(OUT / f"{name}.png")
            degrade(img, i).save(OUT / f"{name}_scan.png")
            gt = pg.gt()
            gt["lang"] = lang
            for variant in (name, f"{name}_scan"):
                (OUT / "gt" / f"{variant}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1))
            print("wrote", name)
        browser.close()


if __name__ == "__main__":
    main()
