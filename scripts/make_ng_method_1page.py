"""Generate docs/PCPG_NG_method_1page.pdf — one page on HOW the natural-gradient
question was measured (method, criterion, validation), with the result as conclusion.

Numbers are literals from results/ng_probe_rerun/, so it regenerates without
re-running anything. Asserts the output is exactly one page.

    python scripts/make_ng_method_1page.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Table, TableStyle)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "PCPG_NG_method_1page.pdf"
FONTS = Path("/usr/share/fonts/truetype/dejavu")

INK = colors.HexColor("#1A1D24")
MUTED = colors.HexColor("#5B6472")
RULE = colors.HexColor("#C9CFD8")
ACCENT = colors.HexColor("#2F5B7C")
BAND = colors.HexColor("#EEF1F5")


def register_fonts():
    for name, f in [("DJSerif", "DejaVuSerif.ttf"), ("DJSerif-Bold", "DejaVuSerif-Bold.ttf"),
                    ("DJSans-Bold", "DejaVuSans-Bold.ttf"), ("DJMono", "DejaVuSansMono.ttf"),
                    ("DJMono-Bold", "DejaVuSansMono-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, FONTS / f))
    # Required, or <b> silently renders as the normal face.
    pdfmetrics.registerFontFamily("DJSerif", normal="DJSerif", bold="DJSerif-Bold",
                                  italic="DJSerif", boldItalic="DJSerif-Bold")


def build():
    register_fonts()
    S = {
        "title": ParagraphStyle("t", fontName="DJSans-Bold", fontSize=14.5, leading=17.5,
                                textColor=INK, alignment=TA_LEFT, spaceAfter=1),
        "sub": ParagraphStyle("s", fontName="DJSerif", fontSize=8.4, leading=11.6,
                              textColor=MUTED, spaceAfter=6),
        "h": ParagraphStyle("h", fontName="DJSans-Bold", fontSize=9.4, leading=12,
                            textColor=ACCENT, spaceBefore=7, spaceAfter=2.5),
        "b": ParagraphStyle("b", fontName="DJSerif", fontSize=8.5, leading=11.9,
                            textColor=INK, spaceAfter=4),
        "cap": ParagraphStyle("c", fontName="DJSerif", fontSize=7.4, leading=10.2,
                              textColor=MUTED, spaceBefore=1.5, spaceAfter=4),
        "cell": ParagraphStyle("cl", fontName="DJSerif", fontSize=7.3, leading=9.6,
                               textColor=INK),
    }
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=13 * mm, bottomMargin=12 * mm,
        title="How we measured whether the PCPG update is a natural gradient",
        author="PCPG project")
    W = doc.width
    F = []
    P = lambda t, st="b": F.append(Paragraph(t, S[st]))

    def tbl(data, widths, right_from=1, bold_cols=(), band_rows=(), wrap_cols=()):
        rows = [data[0]] + [
            [Paragraph(c, S["cell"]) if i in wrap_cols else c for i, c in enumerate(r)]
            for r in data[1:]]
        t = Table(rows, colWidths=widths, hAlign="LEFT")
        st = [("FONT", (0, 0), (-1, 0), "DJSans-Bold", 6.9),
              ("FONT", (0, 1), (-1, -1), "DJMono", 7.4),
              ("FONT", (0, 1), (0, -1), "DJSerif", 7.6),
              ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
              ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
              ("LINEBELOW", (0, -1), (-1, -1), 0.45, RULE),
              ("ALIGN", (right_from, 0), (-1, -1), "RIGHT"),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 2.1),
              ("BOTTOMPADDING", (0, 0), (-1, -1), 2.1),
              ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        for c in bold_cols:
            st.append(("FONT", (c, 1), (c, -1), "DJMono-Bold", 7.4))
        for r in band_rows:
            st.append(("BACKGROUND", (0, r), (-1, r), BAND))
        t.setStyle(TableStyle(st))
        return t

    # ------------------------------------------------------------------ head
    P("How we measured whether the PCPG update is a natural gradient", "title")
    P("Method note &middot; branch <font name='DJMono'>fixing-pcpg</font> &middot; "
      "probe: <font name='DJMono'>scripts/probe_natural_gradient.py</font> (CPU, no "
      "training) &middot; full report: <font name='DJMono'>docs/PCPG_NG_experiments.pdf"
      "</font>", "sub")

    # ------------------------------------------------------------------ 1
    P("1. Turning the claim into something falsifiable", "h")
    P("The natural gradient is F⁻¹∇L with F = Jᵀ F_out J, where J = ∂out/∂θ is the "
      "network Jacobian and F_out = diag(1/σ², 2) is the Gaussian policy's output "
      "Fisher. It needs <b>two</b> ingredients, and PC is only responsible for one:", "b")
    P("• the <b>output metric</b> F_out⁻¹ — supplied, or not, by how the PC target is "
      "constructed (the <font name='DJMono'>natural_target</font> flag);<br/>"
      "• the <b>network factor</b> — could only come from <b>inference</b>, i.e. from "
      "how settled activities reshape the mapping of output errors into weights.", "b")
    P("So &ldquo;PCPG has a natural gradient&rdquo; splits into two separately "
      "measurable sub-claims. Inference is on the hook for the second one only.", "b")

    # ------------------------------------------------------------------ 2
    P("2. What is computed", "h")
    P("Freeze one policy θ and one batch. On that same frozen state, compute the "
      "<b>actual</b> PC update and several reference directions, then compare them as "
      "directions in parameter space by cosine. Nothing is trained.", "b")
    d = [["symbol", "what it is", "the hypothesis it stands for"],
         ["Δ(t1)", "−∂E/∂θ at activities settled for time t1", "what PCPG actually does"],
         ["d_SGD", "−∇L", "vanilla policy gradient"],
         ["d_OUT", "−Jᵀ P u,  P = [σ², ½]", "output metric only — what the natural target encodes"],
         ["d_NG(λ)", "−(F+λI)⁻¹∇L", "the natural gradient — both ingredients"],
         ["d_EQ", "−∇θ[½ rᵀS⁻¹r], S = I + Σ B Bᵀ", "the equilibrium / trust-region prediction"]]
    F.append(tbl(d, [0.11 * W, 0.37 * W, 0.52 * W], right_from=3, wrap_cols=(2,)))
    P("F is applied matrix-free by jvp/vjp plus CG, in exact GGN form, so there is no "
      "action-sampling noise; the damping λ is swept and the best-fit value is part of "
      "the measurement. Two metrics are reported: the ordinary cosine, and the "
      "<b>Fisher-metric cosine</b> cos_F, which is parameterisation-invariant and "
      "equals 1 exactly when a direction buys the most loss improvement per unit KL — "
      "so it cannot be gamed by rescaling.", "cap")

    # ------------------------------------------------------------------ 3
    P("3. The criterion — what would count as a yes", "h")
    P("Inference time t1 is swept from 0 to 4096. <b>t1 = 0 means no inference has "
      "happened at all</b>, so the sweep isolates exactly what inference contributes. "
      "For PC to supply the network factor, two things must hold:", "b")
    P("• cos(Δ, d_NG) must <b>rise</b> with t1 — settling should move the update "
      "<i>onto</i> the natural gradient;<br/>"
      "• the ceiling cos(d_EQ, d_NG) must be high — the thing PC converges to has to "
      "be NG-like in the first place.", "b")
    P("If Δ never leaves its t1 = 0 value, inference contributes nothing and any "
      "natural-gradient content is the target's preconditioning and nothing more.", "b")

    # ------------------------------------------------------------------ 4
    P("4. Validating the instrument first", "h")
    P("At t1 = 0 the update is provably final-layer-only (every hidden error is "
      "identically zero), and its final-layer block must equal d_BP's exactly. Both "
      "come out at <font name='DJMono'><b>1.000000</b></font> to float precision in "
      "every cell. That is the control which says the harness is wired correctly "
      "before any conclusion is read off it.", "b")

    # ------------------------------------------------------------------ 5
    P("5. The answer the measurement gives: no", "h")
    d = [["regime, target family", "cos(Δ, d_NG*)", "cos_F(Δ, d_NG*)", "|Δ| growth", "ceiling cos(d_EQ, d_NG)"],
         ["mixed, Euclidean", "0.75 → 0.20", "0.94 → 0.38", "×3.5", "0.93"],
         ["floor, Euclidean", "0.62 → 0.12", "0.90 → 0.28", "×4.2", "0.91"],
         ["mixed, natural", "0.62 → 0.62", "0.72 → 0.73", "×1.0", "0.70"],
         ["floor, natural", "0.08 → 0.10", "0.54 → 0.56", "×0.8", "0.09"]]
    F.append(tbl(d, [0.24 * W, 0.19 * W, 0.19 * W, 0.14 * W, 0.24 * W], bold_cols=(1,)))
    P("Arrows are t1 = 20 (the production operating point) → t1 = 4096 (equilibrium). "
      "Alignment <b>falls</b> in every cell, so the first criterion fails: the "
      "best-aligned direction PCPG ever produces is the one at t1 ≈ 0, before "
      "inference. The final-layer share stays ≥ 0.92 even at equilibrium, so the "
      "update remains a final-layer delta rule. And the natural family — the one whose "
      "<i>target</i> carries F_out⁻¹ — is the only one that stays aligned, which "
      "locates the natural-gradient content in the target rather than in PC.", "cap")

    # ------------------------------------------------------------------ scope
    P("What this measurement does and does not settle", "h")
    P("It measures <b>natural-gradient content</b>, on initialisation geometry, at "
      "n = 256, single seed, one architecture family (17 → 64 → 12, tanh). It does "
      "<b>not</b> test the adaptive trust-region / saddle-escape property: d_EQ encodes "
      "a local direction-rescaling reading and goes inert exactly at a saddle, so that "
      "claim needs a different instrument. A separate measurement also found production "
      "inference ran at t1/N ≈ 0.01 — about 1% settled — so before that was fixed the "
      "equilibrium end of this sweep was never reached in real runs.", "cap")

    doc.build(F)
    # A method note that silently spills onto page 2 has failed at its one job.
    from pypdf import PdfReader
    n = len(PdfReader(str(OUT)).pages)
    assert n == 1, f"must be exactly one page, got {n}"
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size/1024:.0f} KB, {n} page)")


if __name__ == "__main__":
    build()
