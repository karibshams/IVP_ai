import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from core import InvoicePipeline

load_dotenv()

st.set_page_config(page_title="Invoice Voucher Processor", layout="wide")
st.title("Invoice Voucher Processing System")

if "vouchers" not in st.session_state:
    st.session_state.vouchers = []

uploaded_files = st.file_uploader("Upload vendor invoice PDFs", type="pdf", accept_multiple_files=True)

if st.button("Process Invoices") and uploaded_files:
    os.makedirs("uploads", exist_ok=True)
    file_paths = []
    for f in uploaded_files:
        path = os.path.join("uploads", f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        file_paths.append(path)

    pipeline = InvoicePipeline(api_key=os.getenv("OPENAI_API_KEY"), config_path="gl_config.json")
    st.session_state.pipeline = pipeline
    st.session_state.vouchers = pipeline.run(file_paths, output_dir="outputs")

for idx, v in enumerate(st.session_state.vouchers):
    st.subheader(f"{v.vendor} — {v.company}")
    df = pd.DataFrame([{
        "Invoice #": i.invoice_number,
        "Date": i.invoice_date,
        "Amount": i.invoice_amount,
        "HST": i.hst,
        "GL Code": i.gl_code,
        "GL Description": i.gl_description,
        "Profit Centre": i.profit_centre_code,
        "Net": i.net_amount,
        "Confidence": i.confidence,
        "Flags": ", ".join(i.flags) if i.flags else "-",
    } for i in v.invoices])

    edited = st.data_editor(df, key=f"editor_{idx}", num_rows="fixed")

    if st.button(f"Save Corrections & Generate Voucher PDF - {v.vendor}", key=f"gen_{idx}"):
        for i, row in zip(v.invoices, edited.to_dict("records")):
            i.invoice_number = row["Invoice #"]
            i.invoice_date = row["Date"]
            i.invoice_amount = float(row["Amount"])
            i.hst = float(row["HST"])
            i.gl_code = row["GL Code"]
            i.gl_description = row["GL Description"]
            i.profit_centre_code = row["Profit Centre"]
            i.net_amount = float(row["Net"])

        safe_name = f"{v.vendor}_{v.company}".replace(" ", "_").replace("/", "_")
        out_path = os.path.join("outputs", f"voucher_{safe_name}.pdf")
        st.session_state.pipeline.pdf_gen.process(v, out_path)
        st.success("Voucher updated with your corrections.")

    st.write(f"**Total Net:** {v.total_net:.2f}  **HST:** {v.total_hst:.2f}  **Total:** {v.total_amount:.2f}")

    safe_name = f"{v.vendor}_{v.company}".replace(" ", "_").replace("/", "_")
    out_path = os.path.join("outputs", f"voucher_{safe_name}.pdf")
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            st.download_button(f"Download Voucher PDF - {v.vendor}", f, file_name=os.path.basename(out_path), key=f"dl_{idx}")