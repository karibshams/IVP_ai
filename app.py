import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from core import InvoicePipeline

load_dotenv()

st.set_page_config(page_title="Invoice Voucher Processor", page_icon="🧾", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 2.5rem;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🧾 Invoice Voucher Processing System")
st.caption("Upload vendor invoices → AI extracts & classifies → review & correct → generate voucher PDF")

if "vouchers" not in st.session_state:
    st.session_state.vouchers = []

with st.container(border=True):
    uploaded_files = st.file_uploader("Upload vendor invoice PDFs", type="pdf", accept_multiple_files=True)
    process_clicked = st.button("Process Invoices", type="primary")

if process_clicked and uploaded_files:
    os.makedirs("uploads", exist_ok=True)
    file_paths = []
    for f in uploaded_files:
        path = os.path.join("uploads", f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        file_paths.append(path)

    with st.spinner(f"Processing {len(file_paths)} invoice(s)..."):
        pipeline = InvoicePipeline(api_key=os.getenv("OPENAI_API_KEY"), config_path="gl_config.json")
        st.session_state.pipeline = pipeline
        st.session_state.vouchers = pipeline.run(file_paths, output_dir="outputs")
    st.success(f"Generated {len(st.session_state.vouchers)} voucher(s) grouped by vendor and company.")

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