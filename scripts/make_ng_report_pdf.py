"""Generate docs/PCPG_NG_experiments.pdf — the natural-gradient measurement report.

Self-contained: all numbers are literals taken from the committed run logs
(results/ng_probe_rerun/, results/settled_ablation/), so the PDF regenerates
byte-stably without re-running anything.

    python scripts/make_ng_report_pdf.py

Uses DejaVu TTFs so Greek (sigma, tau, Delta) and arrows render as real glyphs;
ReportLab's built-in Helvetica/Times are WinAnsi-encoded and would draw boxes.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "PCPG_NG_experiments.pdf"
FONTS = Path("/usr/share/fonts/truetype/dejavu")

INK = colors.HexColor("#1A1D24")
MUTED = colors.HexColor("#5B6472")
RULE = colors.HexColor("#C9CFD8")
ACCENT = colors.HexColor("#2F5B7C")      # instrument blue
FLAG = colors.HexColor("#A8552F")        # caveat / retraction
BAND = colors.HexColor("#EEF1F5")


def register_fonts():
    pdfmetrics.registerFont(TTFont("DJSerif", FONTS / "DejaVuSerif.ttf"))
    pdfmetrics.registerFont(TTFont("DJSerif-Bold", FONTS / "DejaVuSerif-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("DJSans-Bold", FONTS / "DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("DJMono", FONTS / "DejaVuSansMono.ttf"))
    pdfmetrics.registerFont(TTFont("DJMono-Bold", FONTS / "DejaVuSansMono-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("DJMono-Oblique", FONTS / "DejaVuSansMono-Oblique.ttf"))
    # Without a family mapping, <b> and <i> inside Paragraph markup silently render
    # as the normal face. DejaVu ships no serif italic, so italic maps to the
    # oblique mono only where that is the base font; serif emphasis uses bold.
    pdfmetrics.registerFontFamily(
        "DJSerif", normal="DJSerif", bold="DJSerif-Bold",
        italic="DJSerif", boldItalic="DJSerif-Bold")
    pdfmetrics.registerFontFamily(
        "DJMono", normal="DJMono", bold="DJMono-Bold",
        italic="DJMono-Oblique", boldItalic="DJMono-Bold")


def styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("title", parent=ss["Title"], fontName="DJSans-Bold",
                                fontSize=19, leading=24, textColor=INK,
                                alignment=TA_LEFT, spaceAfter=2)
    s["sub"] = ParagraphStyle("sub", fontName="DJSerif", fontSize=10.5, leading=15,
                              textColor=MUTED, spaceAfter=10)
    s["h1"] = ParagraphStyle("h1", fontName="DJSans-Bold", fontSize=13, leading=17,
                             textColor=INK, spaceBefore=15, spaceAfter=5)
    s["h2"] = ParagraphStyle("h2", fontName="DJSans-Bold", fontSize=10.5, leading=14,
                             textColor=ACCENT, spaceBefore=10, spaceAfter=3)
    s["body"] = ParagraphStyle("body", fontName="DJSerif", fontSize=9.6, leading=14.4,
                               textColor=INK, spaceAfter=6)
    s["mono"] = ParagraphStyle("mono", fontName="DJMono", fontSize=8.4, leading=12,
                               textColor=INK, leftIndent=8, spaceAfter=6)
    s["cap"] = ParagraphStyle("cap", fontName="DJSerif", fontSize=8.4, leading=11.5,
                              textColor=MUTED, spaceBefore=2, spaceAfter=9)
    s["flag"] = ParagraphStyle("flag", fontName="DJSerif", fontSize=9.4, leading=13.6,
                               textColor=INK, leftIndent=8, rightIndent=6,
                               borderColor=FLAG, borderWidth=0, spaceAfter=7)
    s["cell"] = ParagraphStyle("cell", fontName="DJSerif", fontSize=8.3, leading=11,
                               textColor=INK)
    return s


def table(data, widths, align_right_from=1, highlight_rows=(), highlight_cols=()):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [
        ("FONT", (0, 0), (-1, 0), "DJSans-Bold", 7.8),
        ("FONT", (0, 1), (-1, -1), "DJMono", 8.2),
        ("FONT", (0, 1), (0, -1), "DJSerif", 8.4),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for r in highlight_rows:
        st += [("BACKGROUND", (0, r), (-1, r), BAND),
               ("FONT", (0, r), (-1, r), "DJMono-Bold", 8.2),
               ("FONT", (0, r), (0, r), "DJSerif", 8.4)]
    for c in highlight_cols:
        st.append(("FONT", (c, 1), (c, -1), "DJMono-Bold", 8.2))
    t.setStyle(TableStyle(st))
    return t


def rule():
    t = Table([[""]], colWidths=[170 * mm], rowHeights=[0.7])
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.7, RULE)]))
    return t


def build():
    register_fonts()
    s = styles()
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title="Does the PCPG update implement a natural gradient?",
        author="PCPG project", subject="Natural-gradient measurement report",
    )
    W = doc.width
    F = []
    P = lambda txt, st="body": F.append(Paragraph(txt, s[st]))

    # ---------------------------------------------------------------- title
    P("Does the PCPG update implement a natural gradient?", "title")
    P("Measurement report &middot; branch <font name='DJMono'>fixing-pcpg</font> "
      "&middot; probes run on CPU, jax 0.4.38 / jpc 1.0.0 / equinox 0.13.8 / "
      "diffrax 0.7.2", "sub")

    P("<b>Answer: no.</b> Inference does not move the PC update toward the natural "
      "gradient — at equilibrium it moves it away. The only natural-gradient content "
      "present comes from the target construction (the <font name='DJMono'>"
      "natural_target</font> flag), not from predictive coding. Separately, the "
      "adaptive trust-region rescaling that PC is expected to supply is "
      "<i>numerically inert</i> in this architecture: it changes step length, not "
      "direction.", "body")
    P("Two findings qualify that. First, production inference was running at ~1% of "
      "the integration needed to settle, so until it was fixed the equilibrium claims "
      "were being tested on activities that had barely moved. Second, once settling "
      "was fixed and actually ablated, it <i>did</i> change learning on the bandit — "
      "so the &ldquo;inference is inert&rdquo; result is specific to the geometry it "
      "was measured on and does not generalise.", "body")

    # ---------------------------------------------------------------- what is measured
    F.append(rule())
    P("1. What is actually being compared", "h1")
    P("The probe freezes one policy and one batch, then computes the PC weight update "
      "and several reference directions on that same frozen state. Everything is a "
      "direction in parameter space, so the comparisons are cosines. No training is "
      "involved.", "body")

    refs = [
        ["direction", "definition", "the hypothesis it stands for"],
        ["Δ(t1)", "−∂E/∂θ at settled activities", "what PCPG actually does"],
        ["d_SGD", "−∇L", "vanilla policy gradient"],
        ["d_OUT", "−Jᵀ P u,  P = [σ², ½]", "output metric only — what the natural target encodes"],
        ["d_NG(λ)", "−(F+λI)⁻¹∇L,  F = Jᵀ F_out J", "the natural gradient (both factors)"],
        ["d_BP", "−∇θ E (backprop on the energy)", "&ldquo;PC ≈ backprop&rdquo;"],
        ["d_EQ", "−∇θ[½ rᵀS⁻¹r],  S = I + Σ B Bᵀ", "the equilibrium / trust-region prediction"],
    ]
    body_cells = [[Paragraph(c, s["cell"]) if i == 2 else c for i, c in enumerate(row)]
                  for row in refs[1:]]
    F.append(table([refs[0]] + body_cells, [0.13 * W, 0.35 * W, 0.52 * W],
                   align_right_from=3))
    P("F is the exact Fisher of the policy, applied matrix-free by jvp/vjp plus CG, so "
      "there is no action-sampling noise; λ is swept and the best-fit value is part of "
      "the measurement. F_out = diag(1/σ², 2) is the Gaussian output Fisher. Alongside "
      "the ordinary cosine we report the <b>Fisher-metric cosine</b> cos_F, which is "
      "parameterisation-invariant and equals 1 exactly when a direction buys the most "
      "loss improvement per unit KL.", "cap")

    P("Setup and the harness check", "h2")
    P("Policy is <font name='DJMono'>jpc.make_mlp</font> 17 → 64 → 12 (tanh), batch "
      "n = 256 drawn on-policy, advantages standardised, seed 0. Four <i>regimes</i> "
      "place the σ head, and two <i>target families</i> are run in each: Euclidean "
      "(mean offset ts·A·(z−μ)/σ², the production default) and natural (ts·A·(z−μ), "
      "the 1/σ² amplifier removed). Inference time t1 is swept from 0 to 4096.", "body")
    P("The harness has a built-in anchor: at t1 = 0 no inference has happened, so the "
      "update must be exactly final-layer-only and its final-layer block must equal "
      "d_BP. Both come out at <font name='DJMono'>1.000000</font> to float precision "
      "in every cell, which is what licenses reading the rest.", "body")

    # ---------------------------------------------------------------- exp 1
    F.append(rule())
    P("2. Reproduction of the earlier pilot", "h1")
    P("Findings F1–F5 predate this work. They reproduce on an independent machine, so "
      "what follows builds on a checked base rather than a single run.", "body")
    d = [["quantity", "this run", "pilot"],
         ["floor / Euclidean  cos(Δ, d_NG*), t1=20 → eq", "0.616 → 0.124", "0.68 → 0.11"],
         ["floor / Euclidean  cos_F(Δ, d_NG*), t1=20 → eq", "0.904 → 0.281", "0.93 → 0.27"],
         ["mixed / Euclidean  cos(Δ, d_NG*) at equilibrium", "0.202", "0.20"],
         ["mixed / natural    cos(Δ, d_NG*) at equilibrium", "0.618", "0.60"],
         ["harness anchors at t1 = 0", "1.000000", "1.000000"]]
    F.append(table(d, [0.56 * W, 0.24 * W, 0.20 * W]))
    P("Read the arrows as t1 = 20 (the production operating point) → t1 = 4096 "
      "(equilibrium). Alignment with the natural gradient <i>falls</i> as inference "
      "proceeds, in every cell. That is the core negative result: settling does not "
      "buy natural-gradient geometry, it spends it.", "cap")

    # ---------------------------------------------------------------- exp 2
    P("3. New: the fixed-σ control", "h1")
    P("A fourth regime was added as a control: zero the log_std rows of the final "
      "weight so σ = exp(bias) no longer depends on the state, with bias = 0 giving "
      "σ ≡ 1. This is PPO's parameterisation. It answers &ldquo;how much geometry is "
      "at stake at all?&rdquo;", "body")
    d = [["regime", "σ range", "cos(d_SGD, d_NG)", "cos(d_OUT, d_NG)"],
         ["fixed (control)", "1.000 – 1.000", "0.941", "0.933"],
         ["init", "0.419 – 1.649", "0.925", "0.850"],
         ["mixed", "0.135 – 1.162", "0.622", "0.723"],
         ["floor", "0.135 – 0.297", "0.341", "0.500"]]
    F.append(table(d, [0.26 * W, 0.24 * W, 0.25 * W, 0.25 * W], highlight_rows=(1,)))
    P("With σ constant, F_out = diag(1, 2) is nearly isotropic, so the natural "
      "gradient and the vanilla gradient nearly coincide (0.941). The claim "
      "&ldquo;PC ≈ NG&rdquo; is close to unfalsifiable there, because almost any "
      "direction is ≈ NG. Heteroscedastic σ is what <i>creates</i> the gap the claim "
      "is about, and it is widest at the σ floor (0.341).", "cap")
    P("<b>Consequence for how this gets written up:</b> the fixed-σ regime is the "
      "right control and the wrong test case. It calibrates how much geometry is at "
      "stake; it is never evidence that the update is natural.", "body")

    # ---------------------------------------------------------------- exp 3
    P("4. New: the log_std clamp binds through the target, not through σ", "h1")
    P("The earlier F5 attributed a loss of target fidelity to σ being pinned at its "
      "floor. The control shows that mechanism is at best secondary. In the fixed "
      "regime σ ≡ 1 and <b>0.0%</b> of raw log_std outputs are out of bounds, yet "
      "fidelity on the log_std half is still 0.6995 (Euclidean) / 0.7902 (natural), "
      "not 1.0.", "body")
    P("The reason is that the clip applies to log_scale + offset, so a target leaves "
      "the 2.5-wide window whenever the <i>offset</i> is large — wherever σ sits. "
      "Measured at σ ≡ 1:", "body")
    d = [["target_scale", "log_std targets clipped", "median |offset|"],
         ["1.0  (probe default)", "27.3%", "0.43"],
         ["10.0  (bench ts10)", "80.9%", "4.27"]]
    F.append(table(d, [0.38 * W, 0.32 * W, 0.30 * W], highlight_rows=(2,)))
    P("Every bench config uses ts10, so roughly four in five log_std targets are "
      "truncated in production — in the configuration that was supposed to be the "
      "clean one. Any statement about what the log_std channel encodes has to be "
      "conditioned on target_scale, not only on σ.", "cap")

    F.append(PageBreak())

    # ---------------------------------------------------------------- exp 4
    P("5. New: why the trust-region prediction yields no natural gradient", "h1")
    P("Two separate things are true here, and conflating them is easy.", "body")

    P("5a. The theorem and &ldquo;natural gradient&rdquo; are different claims", "h2")
    P("S = I + Σ_l B_l B_lᵀ with B_l = ∂out/∂z_l is built from the <b>network "
      "Jacobian</b> — a property of the architecture. The Fisher F = Jᵀ F_out J is "
      "built from the <b>output distribution's</b> KL geometry — a property of the "
      "policy. Different objects. Nothing inside PC ever sees the policy "
      "distribution; PC receives a regression target. So distributional geometry has "
      "to be injected at the output, which is exactly what the natural target does.", "body")

    P("5b. And in this architecture S is inert", "h2")
    P("The existing probe reported cos(Δ, d_EQ) but never how far d_EQ is from plain "
      "backprop — which is the entire content of the trust-region claim. Measuring "
      "that directly:", "body")
    d = [["regime", "hidden layers", "eig(S)", "cond(S)", "cos(d_BP, d_EQ)", "cos(d_EQ, d_NG)"],
         ["mixed", "1  (bench)", "1.04 – 1.6", "1.5", "0.9948", "0.931"],
         ["mixed", "2", "1.11 – 1.8", "1.6", "0.9936", "0.921"],
         ["mixed", "4", "1.18 – 1.9", "1.6", "0.9921", "0.912"],
         ["mixed", "8", "1.18 – 2.0", "1.7", "0.9853", "0.897"],
         ["fixed", "1", "1.00 – 1.5", "1.5", "0.9930", "0.632"],
         ["fixed", "4", "1.00 – 1.7", "1.7", "0.9903", "0.278"],
         ["fixed", "8", "1.00 – 1.8", "1.8", "0.9949", "0.076"]]
    F.append(table(d, [0.13 * W, 0.19 * W, 0.17 * W, 0.13 * W, 0.20 * W, 0.18 * W],
                   highlight_cols=(4,)))
    P("cos(d_BP, d_EQ) ≈ 0.99 everywhere: PC-at-equilibrium is <i>directionally</i> "
      "backprop. The cause is the condition number, not the magnitude — a "
      "preconditioner only rotates a gradient if it is anisotropic, and cond(S) ≈ 1.5 "
      "is nearly isotropic, so S rescales step length and leaves direction alone.", "cap")
    P("Both obvious escapes are closed. <b>Depth does not rescue it</b> — the sum runs "
      "over hidden layers, so growth was expected, but cond creeps only 1.5 → 1.8 "
      "from 1 to 8 hidden layers. <b>It is not a tanh-linearisation artifact</b> — in "
      "a linear network, where the equilibrium result is exact rather than "
      "first-order, cos(d_BP, d_EQ) is still 0.985–0.995.", "body")

    P("An important limit on this instrument", "h2")
    P("Innocenti et al. (arXiv:2305.18188) state the headline prediction as <i>PC "
      "escapes saddle points faster than backprop</i>, with the dynamics interpolating "
      "between backprop's gradient direction and a trust-region direction. Two things "
      "follow. Backprop is one endpoint of that interpolation, so cos(Δ, d_BP) ≈ 1 is "
      "the theory's own degenerate case rather than a refutation. And d_EQ goes inert "
      "<i>at</i> a saddle: B_l is a product of weight matrices, so it vanishes as the "
      "weights approach the origin saddle of a linear net. Scaling every weight matrix "
      "by ε in a depth-5 linear net:", "flag")
    d = [["ε", "cond(S)", "cos(d_BP, d_EQ)"],
         ["1.00", "1.62", "0.99097"],
         ["0.30", "1.04", "0.99995"],
         ["0.10", "1.00", "1.00000"],
         ["0.01", "1.00", "1.00000"]]
    F.append(table(d, [0.20 * W, 0.22 * W, 0.28 * W]))
    P("So the rescaling is <i>most</i> inert exactly where the paper predicts the "
      "<i>largest</i> effect. d_EQ encodes a local direction-rescaling reading of the "
      "theorem; saddle escape is a claim about the landscape and the dynamics. These "
      "results should be read as &ldquo;inference supplies no natural-gradient or "
      "output-metric content&rdquo;, which is what they measure, and <b>not</b> as "
      "evidence against the trust-region result. Testing that needs a saddle-escape "
      "experiment under plain gradient descent, which does not exist yet. (The paper's "
      "full text was not retrievable from the run environment, so this rests on its "
      "title and abstract.)", "cap")

    # ---------------------------------------------------------------- exp 5
    F.append(rule())
    P("6. The finding that undercut all of the above: inference was never settling", "h1")
    P("jpc defines the PC energy as a batch <i>mean</i>, F = (1/2N) Σ_i ‖·‖². Each "
      "sample's activities appear only in their own term, so ∂F/∂z_i carries a factor "
      "1/N and every activity's velocity is divided by N. Because dividing a gradient "
      "flow by a positive constant cannot move its fixed point, this is purely a "
      "change of <i>clock rate</i> — but it means max_t1 is an ODE end time whose "
      "effective per-sample time is τ = t1/N.", "body")
    P("Measured: the residual is a function of τ <b>alone</b> (N = 64 and N = 256 "
      "agree to three decimals), so one curve covers every batch size.", "body")
    d = [["τ = t1/N", "residual / initial", ""],
         ["0.00977", "0.988", "← production (max_t1=20, N=2048): 1.2% settled"],
         ["0.3", "0.645", ""],
         ["1.0", "0.218", ""],
         ["3.0", "0.026", "90% settled"],
         ["10.0", "0.002", "← converged; τ=30 and τ=100 do not improve"]]
    body_cells = [[r[0], r[1], Paragraph(r[2], s["cell"])] for r in d[1:]]
    F.append(table([d[0]] + body_cells, [0.18 * W, 0.22 * W, 0.60 * W],
                   align_right_from=1, highlight_rows=(1, 5)))
    P("Production therefore removed 1.2% of the residual where τ ≈ 10 was needed — a "
      "~1000× shortfall. Raising max_t1 cannot fix it: jpc hard-codes a stop at "
      "t = 4096, which is τ = 2 (93.8% settled) at bench N, while convergence needs "
      "t1 ≈ 20,480. A second bug compounds it — jpc's steady-state early-stop tests "
      "rms_norm(activities) &lt; ~1e−3 rather than rms_norm(dz/dt), so it never fires "
      "and inference simply runs the clock out.", "cap")
    P("The fix, and why it is safe", "h2")
    P("Multiply the inference vector field by N, cancelling the 1/N, so max_t1 <i>is</i> "
      "the per-sample time. The weight step keeps the 1/N, so gradients remain a batch "
      "mean and learning rates carry over unchanged. Verified two ways: the settled "
      "activities agree with jpc integrated to t1 = 40N at cos = 1.000000 (max relative "
      "difference ~6e−4), confirming the fixed point is untouched; and at the real "
      "bench N = 2048 the residual goes 0.988 → 0.0037. Cost is not a factor — "
      "0.199 s/update against 0.221–0.254 s before — because past equilibrium the "
      "adaptive solver grows its step.", "body")

    # ---------------------------------------------------------------- exp 6
    P("7. The ablation this unblocked, and what it overturned", "h1")
    P("Bandit, favor_suboptimal init (π₀ = 0.018), 60k steps, seeds 1–5, both PC "
      "algorithms, arms paired by seed; 20/20 runs completed. The arms are residual "
      "0.90 (unsettled) against 0.003 (settled).", "body")
    P("Final performance shows nothing — but it cannot: both arms reach 5/5 success at "
      "π ≈ 0.98, i.e. the task is saturated at this budget.", "body")
    d = [["algorithm", "unsettled", "settled", "paired Δ", "verdict"],
         ["pc_reinforce", "0.989 ± 0.003", "0.987 ± 0.003", "−0.002 ± 0.002", "n.s."],
         ["pc_actor_critic", "0.992 ± 0.003", "0.977 ± 0.008", "−0.015 ± 0.007", "n.s."]]
    F.append(table(d, [0.24 * W, 0.19 * W, 0.19 * W, 0.20 * W, 0.18 * W]))
    P("Steps-to-threshold uses the whole learning curve instead of its endpoint, and "
      "is unambiguous — settling is <b>slower</b>:", "body")
    d = [["algorithm", "paired Δ steps to π ≥ 0.9", "t(4)", "95% CI"],
         ["pc_reinforce", "+3661 ± 407", "+9.00", "[+2532, +4790]"],
         ["pc_actor_critic", "+5085 ± 1702", "+2.99", "[+361, +9809]"]]
    F.append(table(d, [0.26 * W, 0.30 * W, 0.16 * W, 0.28 * W], highlight_cols=(1,)))
    P("All five pc_reinforce seeds slower; four of five pc_actor_critic slower with "
      "one tie. t_crit at n = 5 is 2.776, so both clear it. (A 2·SEM rule would have "
      "mislabelled the final-π result as significant; these use a paired t-test.)", "cap")
    P("This falsifies the §5b prediction on this task — but does not overturn §5b. "
      "Measured directly in the <i>bandit's</i> geometry (discrete softmax, 1 → 32 → "
      "2), settled against unsettled gives cos = 0.67–0.89 (mean ≈ 0.78) and a norm "
      "ratio of 1.05–1.65: inference rotates the update substantially here, unlike the "
      "Gaussian 17 → 64 → 12 geometry where cos(d_BP, d_EQ) ≈ 0.99. And the slowdown "
      "is not a step-size artifact, since settled steps are <i>larger</i> yet converge "
      "later. The lesson is that <b>&ldquo;inference is inert&rdquo; is specific to "
      "the geometry it was measured on.</b> The MuJoCo/Gaussian case has not been run "
      "— it needs a GPU — and is the open question.", "body")

    # ---------------------------------------------------------------- conclusions
    F.append(rule())
    P("8. What is established, and what is not", "h1")
    P("<b>Established.</b> Alignment with the natural gradient falls monotonically "
      "with inference time in all four regimes; the final-layer share stays ≥ 0.92 "
      "even at equilibrium, so the update is a final-layer delta rule; the only "
      "natural-gradient ingredient present is the target's F_out⁻¹; the trust-region "
      "rescaling S is near-isotropic and directionally inert in this architecture, "
      "including in the linear case where the result is exact; the log_std clamp "
      "truncates ~81% of targets at bench target_scale; and production inference ran "
      "at ~1% settled, now fixed and verified.", "body")
    P("<b>Not established.</b> That PC has no trust-region property — the instrument "
      "used here cannot test the saddle-escape claim, and goes inert precisely at a "
      "saddle. That settling does or does not matter on MuJoCo — it changed the bandit "
      "materially, and the geometry differs. That any of this explains why PC succeeds "
      "on the bandit (0.968 ± 0.029, 9/10, where PPO and REINFORCE reach ~0.02, 0/10) "
      "while failing on continuous control: settling more makes the bandit "
      "<i>slower</i>, so the advantage is not &ldquo;more inference&rdquo;.", "body")

    P("Limitations to state alongside any of these numbers", "h1")
    P("Sections 2–5 are initialisation geometry: weights at init plus σ-head bias "
      "shifts, a synthetic though distributionally matched batch, single seed, "
      "n = 256, one architecture family. Trained checkpoints may sit in "
      "better-conditioned geometry where S is anisotropic — that is the single most "
      "likely way the inertness result softens, and the real-checkpoint probe that "
      "would settle it is specified but unbuilt. The depth sweep varies architecture "
      "but stays at init. cos(d_EQ, d_NG) uses a best-fit damping λ per cell. Section "
      "7 is the only real-training result and is bandit-only, at n = 5.", "body")

    P("Reproducing", "h1")
    P("All CPU, no environment needed:", "body")
    P("python scripts/probe_natural_gradient.py<br/>"
      "python scripts/probe_trust_region_metric.py<br/>"
      "python scripts/probe_trust_region_metric.py --act-fn linear<br/>"
      "python scripts/probe_inference_settling.py<br/>"
      "python scripts/run_settled_ablation.py --seeds 1 2 3 4 5", "mono")
    P("Logs in <font name='DJMono'>results/ng_probe_rerun/</font> and "
      "<font name='DJMono'>results/settled_ablation/</font>. Narrative versions of "
      "these findings are in <font name='DJMono'>docs/PCPG_GEOMETRY_FINDINGS.md</font> "
      "and <font name='DJMono'>docs/PCPG_NATURAL_GRADIENT_PROBES.md</font>.", "cap")

    def footer(canvas, docu):
        canvas.saveState()
        canvas.setFont("DJSerif", 7.6)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, 10 * mm,
                          "PCPG natural-gradient measurement report")
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"{docu.page}")
        canvas.restoreState()

    doc.build(F, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    build()
