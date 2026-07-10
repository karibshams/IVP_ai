import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from core import InvoicePipeline

load_dotenv()

st.set_page_config(page_title="Invoice Voucher AI", page_icon="🧾", layout="wide")

CARD_COLORS = [
    ("#4F46E5", "#6366F1"),  # blue/indigo
    ("#F59E0B", "#FBBF24"),  # orange
    ("#A21CAF", "#C026D3"),  # purple/magenta
    ("#DC2626", "#EF4444"),  # red
]

st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1300px;}
    #MainMenu, footer {visibility: hidden;}

    .app-header {display: flex; align-items: center; justify-content: space-between;
                 padding-bottom: 1.2rem; margin-bottom: 1.5rem; border-bottom: 1px solid #eee;}
    .app-title {display: flex; align-items: center; gap: 12px;}
    .app-title .logo {background: linear-gradient(135deg,#4F46E5,#A21CAF); width: 44px; height: 44px;
                       border-radius: 12px; display: flex; align-items: center; justify-content: center;
                       font-size: 22px;}
    .app-title h1 {font-size: 1.3rem; margin: 0; font-weight: 700;}
    .app-title p {margin: 0; font-size: 0.8rem; color: #888;}

    .balance-card {border-radius: 16px; padding: 18px 20px; color: white; position: relative;
                   overflow: hidden; min-height: 120px;}
    .balance-card .label {font-size: 0.8rem; opacity: 0.9;}
    .balance-card .amount {font-size: 1.6rem; font-weight: 700; margin: 6px 0 14px 0;}
    .balance-card .meta {display: flex; justify-content: space-between; font-size: 0.7rem; opacity: 0.85;}
    .balance-card .status {position: absolute; top: 14px; right: 16px; font-size: 0.7rem;
                            background: rgba(255,255,255,0.25); padding: 3px 10px; border-radius: 20px;}

    .voucher-panel {background: white; border-radius: 16px; padding: 4px 0 0 0; margin-top: 10px;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div class="app-title">
        <div class="logo">🧾</div>
        <div><h1>Invoice Voucher AI</h1><p>Zulich Group — Automated Voucher Processing</p></div>
    </div>
</div>
""", unsafe_allow_html=True)

if "vouchers" not in st.session_state:
    st.session_state.vouchers = []

with st.sidebar:
    st.markdown("### 🧾 Invoice Voucher AI")
    st.caption("Automated GL & Profit Centre classification")
    st.divider()
    total_vouchers = len(st.session_state.vouchers)
    total_amount = sum(v.total_amount for v in st.session_state.vouchers)
    flagged = sum(1 for v in st.session_state.vouchers for i in v.invoices if i.flags)
    st.metric("Vouchers Generated", total_vouchers)
    st.metric("Total Processed", f"${total_amount:,.2f}")
    st.metric("Flagged for Review", flagged)
    st.divider()
    st.caption("Upload → Extract → Classify → Review → Download")

with st.container(border=True):
    uploaded_files = st.file_uploader(
        "📤 Upload vendor invoices — PDF or image (JPG, PNG, TIFF, WEBP, etc.)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp", "gif"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) ready: " + ", ".join(f.name for f in uploaded_files))
    process_clicked = st.button("⚡ Process Invoices", type="primary")

if process_clicked and uploaded_files:
    os.makedirs("uploads", exist_ok=True)
    file_paths = []
    for f in uploaded_files:
        path = os.path.join("uploads", f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        file_paths.append(path)

    progress_bar = st.progress(0, text="Starting...")

    def update_progress(idx, total, filename):
        progress_bar.progress((idx) / total, text=f"Processing {idx + 1}/{total}: {filename}")

    pipeline = InvoicePipeline(api_key=os.getenv("OPENAI_API_KEY"), config_path="gl_config.json")
    st.session_state.pipeline = pipeline
    st.session_state.vouchers = pipeline.run(file_paths, output_dir="outputs", progress_callback=update_progress)
    progress_bar.progress(1.0, text="Done!")
    st.rerun()

if st.session_state.vouchers:
    st.markdown("#### Vouchers")
    cols = st.columns(len(st.session_state.vouchers[:4]) or 1)
    for i, v in enumerate(st.session_state.vouchers[:4]):
        c1, c2 = CARD_COLORS[i % len(CARD_COLORS)]
        needs_review = any(inv.flags for inv in v.invoices)
        status = "⚠️ Review" if needs_review else "✅ Clean"
        with cols[i]:
            st.markdown(f"""
            <div class="balance-card" style="background: linear-gradient(135deg,{c1},{c2});">
                <div class="status">{status}</div>
                <div class="label">{v.vendor}</div>
                <div class="amount">${v.total_amount:,.2f}</div>
                <div class="meta"><span>{v.company}</span><span>{len(v.invoices)} invoice(s)</span></div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

for idx, v in enumerate(st.session_state.vouchers):
    needs_review = any(i.flags for i in v.invoices)
    status = "⚠️ Needs Review" if needs_review else "✅ Clean"

    with st.expander(f"{v.vendor}  ·  {v.company}  ·  {status}", expanded=needs_review):
        c1, c2, c3 = st.columns(3)
        c1.metric("Net Amount", f"${v.total_net:,.2f}")
        c2.metric("HST", f"${v.total_hst:,.2f}")
        c3.metric("Total", f"${v.total_amount:,.2f}")

        df = pd.DataFrame([{
            "Invoice #": i.invoice_number,
            "Date": i.invoice_date,
            "Amount": i.invoice_amount,
            "HST": i.hst,
            "GL Code": i.gl_code,
            "GL Description": i.gl_description,
            "Profit Centre": f"{i.profit_centre_code} (R)" if i.is_rental else i.profit_centre_code,
            "Net": i.net_amount,
            "Confidence": i.confidence,
            "Flags": ", ".join(i.flags) if i.flags else "-",
        } for i in v.invoices])

        st.caption("Review and correct any fields below before generating the final voucher.")
        edited = st.data_editor(df, key=f"editor_{idx}", num_rows="fixed", use_container_width=True)

        col_gen, col_dl = st.columns([1, 1])
        with col_gen:
            if st.button("💾 Save Corrections & Generate PDF", key=f"gen_{idx}", use_container_width=True):
                for i, row in zip(v.invoices, edited.to_dict("records")):
                    i.invoice_number = row["Invoice #"]
                    i.invoice_date = row["Date"]
                    i.invoice_amount = float(row["Amount"])
                    i.hst = float(row["HST"])
                    i.gl_code = row["GL Code"]
                    i.gl_description = row["GL Description"]
                    i.profit_centre_code = str(row["Profit Centre"]).replace(" (R)", "")
                    i.net_amount = float(row["Net"])

                safe_name = f"{v.vendor}_{v.company}".replace(" ", "_").replace("/", "_")
                out_path = os.path.join("outputs", f"voucher_{safe_name}.pdf")
                st.session_state.pipeline.pdf_gen.process(v, out_path)
                st.success("Voucher updated with your corrections.")

        safe_name = f"{v.vendor}_{v.company}".replace(" ", "_").replace("/", "_")
        out_path = os.path.join("outputs", f"voucher_{safe_name}.pdf")
        with col_dl:
            if os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Download Voucher PDF", f, file_name=os.path.basename(out_path),
                                        key=f"dl_{idx}", use_container_width=True)