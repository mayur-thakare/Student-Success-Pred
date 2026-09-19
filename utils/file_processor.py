import pandas as pd
import io


def process_uploaded_file(uploaded_file):

    filename = uploaded_file.filename.lower()

    # Excel
    if filename.endswith(".xlsx"):

        return pd.read_excel(
            uploaded_file,
            engine="openpyxl"
        )

    # CSV
    elif filename.endswith(".csv"):

        return pd.read_csv(uploaded_file)

    # PDF
    elif filename.endswith(".pdf"):

        try:
            import pdfplumber
        except ImportError:
            raise Exception(
                "PDF support requires the pdfplumber package."
            )

        rows = []

        with pdfplumber.open(uploaded_file) as pdf:

            for page in pdf.pages:

                tables = page.extract_tables()

                for table in tables:

                    if not table:
                        continue

                    headers = table[0]

                    for row in table[1:]:

                        if len(row) == len(headers):

                            rows.append(row)

        if not rows:
            raise Exception(
                "No table data could be extracted from the PDF."
            )

        # Extract headers from first table
        with pdfplumber.open(uploaded_file) as pdf:

            first_table = None

            for page in pdf.pages:

                tables = page.extract_tables()

                if tables:
                    first_table = tables[0]
                    break

        if first_table is None:
            raise Exception(
                "No table found in the PDF."
            )

        headers = first_table[0]

        return pd.DataFrame(
            rows,
            columns=headers
        )

    else:

        raise Exception(
            "Unsupported file type. "
            "Please upload CSV, Excel, or PDF."
        )