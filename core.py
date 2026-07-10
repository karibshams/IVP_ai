import os
import io
import json
import base64
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict

import fitz  # PyMuPDF
from openai import OpenAI
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PyPDF2 import PdfReader, PdfWriter

HST_RATE = 0.13


# ---------------- Data Models ----------------

@dataclass
class Invoice:
    file_name: str
    raw_text: str
    vendor: str = ""
    invoice_number: str = ""
    invoice_date: str = ""
    invoice_amount: float = 0.0
    net_amount: float = 0.0
    hst: float = 0.0
    gl_code: str = ""
    gl_description: str = ""
    profit_centre_code: str = ""
    profit_centre_description: str = ""
    is_rental: bool = False
    company: str = ""
    confidence: float = 0.0
    flags: List[str] = field(default_factory=list)
    pdf_path: str = ""


@dataclass
class Voucher:
    vendor: str
    company: str
    invoices: List[Invoice]

    @property
    def total_net(self):
        return sum(i.net_amount for i in self.invoices)

    @property
    def total_hst(self):
        return sum(i.hst for i in self.invoices)

    @property
    def total_amount(self):
        return sum(i.invoice_amount for i in self.invoices)


# ---------------- Config ----------------

class GLConfig:
    def __init__(self, gl_accounts_by_company: dict, gl_chart_map: dict, profit_centres_by_company: dict,
                 rental_rule: dict, excluded_profit_centres: dict):
        self.gl_accounts_by_company = gl_accounts_by_company
        self.gl_chart_map = gl_chart_map
        self.profit_centres_by_company = profit_centres_by_company
        self.rental_rule = rental_rule
        self.excluded_profit_centres = excluded_profit_centres

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            data.get("gl_accounts_by_company", {}),
            data.get("gl_chart_map", {}),
            data.get("profit_centres_by_company", {}),
            data.get("rental_rule", {}),
            data.get("excluded_profit_centres", {}),
        )

    def is_rental(self, company: str, profit_centre_code: str) -> bool:
        if company != self.rental_rule.get("company"):
            return False
        try:
            return int(profit_centre_code) > int(self.rental_rule.get("after_code"))
        except (ValueError, TypeError):
            return False

    def is_excluded(self, company: str, profit_centre_code: str):
        return self.excluded_profit_centres.get(company, {}).get(profit_centre_code)

    def gl_accounts_for(self, company: str) -> dict:
        chart_name = self.gl_chart_map.get(company, "")
        return self.gl_accounts_by_company.get(chart_name, {})


# ---------------- Agents ----------------

class IntakeAgent:
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}

    def process(self, file_path: str) -> Invoice:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._process_pdf(file_path)
        if ext in self.IMAGE_EXTENSIONS:
            return self._process_image(file_path)
        raise ValueError(f"Unsupported file type: {ext}")

    def _process_pdf(self, file_path: str) -> Invoice:
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        if not text.strip():
            text = self._ocr_pdf(doc)
        doc.close()
        return Invoice(file_name=os.path.basename(file_path), raw_text=text, pdf_path=file_path)

    def _process_image(self, file_path: str) -> Invoice:
        from PIL import Image
        import pytesseract
        img = Image.open(file_path).convert("RGB")
        text = pytesseract.image_to_string(img)
        pdf_path = os.path.splitext(file_path)[0] + "_converted.pdf"
        img.save(pdf_path, "PDF")
        return Invoice(file_name=os.path.basename(file_path), raw_text=text, pdf_path=pdf_path)

    def _ocr_pdf(self, doc) -> str:
        from PIL import Image
        import pytesseract
        text = ""
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text += pytesseract.image_to_string(img) + "\n"
        return text


class ExtractorAgent:
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    def process(self, invoice: Invoice) -> Invoice:
        prompt = (
            "Extract these fields from the invoice text as JSON only, no extra text:\n"
            "vendor, invoice_number, invoice_date (YYYY-MM-DD), invoice_amount (number), "
            "net_amount (number, pre-tax), hst (number).\n"
            "If HST is not explicitly stated, calculate it as net_amount * 0.13.\n\n"
            f"Invoice text:\n{invoice.raw_text}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        invoice.vendor = data.get("vendor", "")
        invoice.invoice_number = data.get("invoice_number", "")
        invoice.invoice_date = data.get("invoice_date", "")
        invoice.invoice_amount = float(data.get("invoice_amount") or 0)
        invoice.net_amount = float(data.get("net_amount") or 0)
        invoice.hst = float(data.get("hst") or round(invoice.net_amount * HST_RATE, 2))
        return invoice


class ClassifierAgent:
    def __init__(self, client: OpenAI, config: GLConfig, model: str = "gpt-4o"):
        self.client = client
        self.config = config
        self.model = model

    def process(self, invoice: Invoice) -> Invoice:
        company, company_confidence = self._match_company(invoice)
        invoice.company = company

        pc_code, pc_desc, pc_confidence = self._match_profit_centre(invoice, company)
        invoice.profit_centre_code = pc_code
        invoice.profit_centre_description = pc_desc
        invoice.is_rental = self.config.is_rental(company, pc_code)

        excluded_reason = self.config.is_excluded(company, pc_code)
        if excluded_reason:
            invoice.flags.append(f"excluded_profit_centre: {excluded_reason}")

        gl_code, gl_desc, gl_confidence = self._match_gl_account(invoice, company)
        invoice.gl_code = gl_code
        invoice.gl_description = gl_desc
        invoice.confidence = min(company_confidence, pc_confidence, gl_confidence)
        return invoice

    def _match_company(self, invoice: Invoice):
        companies = list(self.config.profit_centres_by_company.keys())
        prompt = (
            "Given this invoice text, determine which company/entity it belongs to, "
            "based on the billing address, ship-to address, or project mentioned. "
            "Choose only from this list of companies: " + ", ".join(companies) + ".\n"
            "Respond as JSON only: {company, confidence (0-1)}.\n\n"
            f"Invoice text:\n{invoice.raw_text}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("company", ""), float(data.get("confidence") or 0)

    def _match_profit_centre(self, invoice: Invoice, company: str):
        profit_centres = self.config.profit_centres_by_company.get(company, {})
        prompt = (
            "Match this invoice to the correct profit centre from the list below "
            "(project/property address or name). Respond as JSON only: "
            "{profit_centre_code, profit_centre_description, confidence (0-1)}.\n\n"
            f"Profit Centres: {json.dumps(profit_centres)}\n\n"
            f"Invoice text:\n{invoice.raw_text}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return (data.get("profit_centre_code", ""), data.get("profit_centre_description", ""),
                float(data.get("confidence") or 0))

    def _match_gl_account(self, invoice: Invoice, company: str):
        gl_accounts = self.config.gl_accounts_for(company)
        prompt = (
            "Match this invoice's expense type to the correct GL account from the list below. "
            "Respond as JSON only: {gl_code, gl_description, confidence (0-1)}.\n\n"
            f"GL Accounts: {json.dumps(gl_accounts)}\n\n"
            f"Invoice text:\n{invoice.raw_text}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("gl_code", ""), data.get("gl_description", ""), float(data.get("confidence") or 0)


class ValidatorAgent:
    CONFIDENCE_THRESHOLD = 0.85

    def process(self, invoice: Invoice, seen_invoice_numbers: set) -> Invoice:
        if not invoice.vendor:
            invoice.flags.append("missing_vendor")
        if not invoice.gl_code:
            invoice.flags.append("missing_gl_code")
        if not invoice.profit_centre_code:
            invoice.flags.append("missing_profit_centre")
        if not invoice.company:
            invoice.flags.append("missing_company")
        if any(f.startswith("excluded_profit_centre") for f in invoice.flags):
            invoice.flags.append("requires_manual_reassignment")
        if invoice.confidence < self.CONFIDENCE_THRESHOLD:
            invoice.flags.append("low_confidence")
        if invoice.invoice_number in seen_invoice_numbers:
            invoice.flags.append("duplicate_invoice_number")
        expected_hst = round(invoice.net_amount * HST_RATE, 2)
        if abs(invoice.hst - expected_hst) > 0.05:
            invoice.flags.append("hst_mismatch")
        seen_invoice_numbers.add(invoice.invoice_number)
        return invoice


class VoucherBuilderAgent:
    def process(self, invoices: List[Invoice]) -> List[Voucher]:
        groups = defaultdict(list)
        for inv in invoices:
            groups[(inv.vendor, inv.company)].append(inv)
        return [Voucher(vendor=v, company=c, invoices=items) for (v, c), items in groups.items()]


class PDFGeneratorAgent:
    def process(self, voucher: Voucher, output_path: str) -> str:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        width, height = letter
        y = height - 50
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, f"Payment Voucher - {voucher.vendor}")
        y -= 20
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Company: {voucher.company}")
        y -= 30
        headers = ["Invoice #", "Date", "Amount", "HST", "GL Code", "Profit Centre", "Net"]
        c.setFont("Helvetica-Bold", 9)
        for i, h in enumerate(headers):
            c.drawString(50 + i * 75, y, h)
        y -= 15
        c.setFont("Helvetica", 9)
        for inv in voucher.invoices:
            pc_label = f"{inv.profit_centre_code} (R)" if inv.is_rental else inv.profit_centre_code
            row = [inv.invoice_number, inv.invoice_date, f"{inv.invoice_amount:.2f}",
                   f"{inv.hst:.2f}", inv.gl_code, pc_label, f"{inv.net_amount:.2f}"]
            for i, val in enumerate(row):
                c.drawString(50 + i * 75, y, str(val))
            y -= 15
        y -= 10
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, y, f"Total Net: {voucher.total_net:.2f}   HST: {voucher.total_hst:.2f}   Total: {voucher.total_amount:.2f}")
        c.save()
        buf.seek(0)

        writer = PdfWriter()
        voucher_pdf = PdfReader(buf)
        for page in voucher_pdf.pages:
            writer.add_page(page)
        for inv in voucher.invoices:
            src = PdfReader(inv.pdf_path)
            for page in src.pages:
                writer.add_page(page)

        with open(output_path, "wb") as f:
            writer.write(f)
        return output_path


class InvoicePipeline:
    def __init__(self, api_key: str, config_path: str):
        client = OpenAI(api_key=api_key)
        config = GLConfig.load(config_path)
        self.intake = IntakeAgent()
        self.extractor = ExtractorAgent(client)
        self.classifier = ClassifierAgent(client, config)
        self.validator = ValidatorAgent()
        self.builder = VoucherBuilderAgent()
        self.pdf_gen = PDFGeneratorAgent()

    def run(self, file_paths: List[str], output_dir: str, progress_callback=None) -> List[Voucher]:
        os.makedirs(output_dir, exist_ok=True)
        invoices = []
        seen_numbers = set()
        total = len(file_paths)
        for idx, path in enumerate(file_paths):
            if progress_callback:
                progress_callback(idx, total, os.path.basename(path))
            inv = self.intake.process(path)
            inv = self.extractor.process(inv)
            inv = self.classifier.process(inv)
            inv = self.validator.process(inv, seen_numbers)
            invoices.append(inv)

        vouchers = self.builder.process(invoices)
        for v in vouchers:
            safe_name = f"{v.vendor}_{v.company}".replace(" ", "_").replace("/", "_")
            out_path = os.path.join(output_dir, f"voucher_{safe_name}.pdf")
            self.pdf_gen.process(v, out_path)
        return vouchers