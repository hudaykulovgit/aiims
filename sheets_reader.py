import streamlit as st
import pandas as pd
import os
import pickle
from datetime import datetime
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI
from dotenv import load_dotenv
import base64
from email.mime.text import MIMEText

from report_generator import generate_monthly_report

load_dotenv()

# --------------------------- CONFIG ---------------------------
SCOPE = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/documents'
]
SHEET_ID = '1LD9tmCWotdS4X2xh_AXmh4V59COXN1fCwMGlgPV1Sm8'  # 🔁 Replace with your real sheet ID

RANGES = {
    'inventory': 'Inventory!A1:I',
    'email_sent': 'EmailSent!A1:H',
    'log': 'Log!A1:K',  # 11 columns: timestamp, event_type, product_id, notes, status, assistant_action, supplier_email, email_subject, days_to_reply, urgency_flag, ai_summary_snippet
}
def log_event(
    sheets_service,
    event_type,
    product_id=None,
    notes=None,
    status=None,
    assistant_action=None,
    supplier_email=None,
    email_subject=None,
    days_to_reply=None,
    urgency_flag=None,
    ai_summary_snippet=None,
):
    """
    Write a log entry to the Log sheet.
    Columns (order must match sheet): 
    timestamp, event_type, product_id, supplier_email, email_subject, status, days_to_reply, urgency_flag, ai_summary_snippet, assistant_action
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Order: timestamp, event_type, product_id, supplier_email, email_subject, status, days_to_reply, urgency_flag, ai_summary_snippet, assistant_action
    entry = [
        now_str,                        # timestamp
        event_type or "",               # event_type
        product_id or "",               # product_id
        supplier_email or "",           # supplier_email
        email_subject or "",            # email_subject
        status or "",                   # status
        str(days_to_reply) if days_to_reply is not None else "",  # days_to_reply
        urgency_flag or "",             # urgency_flag
        ai_summary_snippet or "",       # ai_summary_snippet
        assistant_action or ""          # assistant_action
    ]
    sheets_service.spreadsheets().values().append(
        spreadsheetId=SHEET_ID,
        range=RANGES['log'],
        valueInputOption='USER_ENTERED',
        insertDataOption='INSERT_ROWS',
        body={'values': [entry]}
    ).execute()
# --------------------------------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Authenticate and create Sheets service
def authenticate_google():
    creds = None
    if os.path.exists('token.json'):
        with open('token.json', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPE)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'wb') as token:
            pickle.dump(creds, token)
    return creds


# ✅ Load a worksheet into a DataFrame
def load_sheet_data(service, range_name):
    result = service.spreadsheets().values().get(
        spreadsheetId=SHEET_ID,
        range=range_name
    ).execute()
    values = result.get('values', [])
    if not values:
        return pd.DataFrame()
    # Pad rows to have the same length as header to avoid errors
    max_len = len(values[0])
    padded_values = [row + [None]*(max_len - len(row)) for row in values[1:]]
    return pd.DataFrame(padded_values, columns=values[0])


def create_message(sender, to, subject, message_text):
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw}


def send_email_via_gmail(service, user_id, message):
    try:
        sent_message = service.users().messages().send(userId=user_id, body=message).execute()
        return sent_message
    except Exception as e:
        raise e


# ✅ Main logic
def main():
    st.title("📦 AI Inventory System Dashboard")
    # Sidebar model selector (tooltip entries for stock categories and reliability score removed)
    model_options = [
        # "google/gemini-2.5-flash-preview-05-20",
        "openai/gpt-4.1-nano",
        # "google/gemini-2.5-flash-preview-05-20",
        # "google/gemini-2.5-pro-preview", 
        "x-ai/grok-3-beta",
        # "deepseek/deepseek-chat-v3-0324:free",
        "anthropic/claude-sonnet-4",
        ]
    selected_model = st.sidebar.selectbox(
        "🤖 Choose AI Model",
        model_options,
        index=0,
        help="Choose which AI model to use for summaries and responses."
    )

    creds = authenticate_google()
    sheets_service = build('sheets', 'v4', credentials=creds)
    gmail_service = build('gmail', 'v1', credentials=creds)
    calendar_service = build('calendar', 'v3', credentials=creds)

    # Load all 3 sheets
    inventory_df = load_sheet_data(sheets_service, RANGES['inventory'])
    email_df = load_sheet_data(sheets_service, RANGES['email_sent'])
    log_df = load_sheet_data(sheets_service, RANGES['log'])

    # 🧼 Convert columns to correct types
    inventory_df['quantity'] = pd.to_numeric(inventory_df['quantity'])
    inventory_df['threshold'] = pd.to_numeric(inventory_df['threshold'])
    inventory_df['price'] = pd.to_numeric(inventory_df['price'])

    # === Stock Status Categorization ===
    def categorize_stock(row):
        try:
            q = float(row['quantity'])
            t = float(row['threshold'])
        except Exception:
            return "Unknown"
        if q < t:
            return "Critical"
        elif t <= q <= t * 1.5:
            return "Healthy"
        elif q > t * 1.5:
            return "Overstocked"
        else:
            return "Unknown"
    inventory_df['stock_status'] = inventory_df.apply(categorize_stock, axis=1)

    tab1, tab2, tab3 = st.tabs(["📦 Inventory & Performance", "📊 Analytics", "📬 Suppliers & Emails"])

    with tab1:
        st.subheader("Inventory Overview")
        # Calculate restock cost only where quantity < threshold, else 0
        restock_amount = (inventory_df['threshold'] - inventory_df['quantity']).clip(lower=0)
        restock_cost = restock_amount * inventory_df['price']
        inventory_df = inventory_df.copy()
        inventory_df['restock_cost'] = restock_cost
        st.dataframe(inventory_df)
        st.caption("ℹ️ **Stock Categories:** Critical = quantity < threshold | Healthy = threshold ≤ quantity ≤ 1.5 × threshold | Overstocked = quantity > 1.5 × threshold")

        total_restock_cost = restock_cost.sum()
        st.metric("💵 Total Restock Cost", f"${total_restock_cost:,.2f}")

        # Low stock table below restock cost
        low_stock_df = inventory_df[inventory_df['quantity'] < inventory_df['threshold']]
        low_stock_count = len(low_stock_df)
        if not low_stock_df.empty:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.subheader("⚠️ Low Stock Items")
            with col2:
                st.markdown(f"**{low_stock_count} Low Stock Items**")

            def highlight_low_stock(row):
                return ['background-color: #ffcccc' if row['quantity'] < row['threshold'] else '' for _ in row]

            # Display DataFrame with red tag for low stock in product name
            display_df = low_stock_df.copy()
            # Fallback for 'name' if not in columns
            if 'product_name' not in display_df.columns:
                display_df['product_name'] = display_df['product_id']
            # Compose a new column with tag if quantity < threshold
            def red_badge(text):
                # Returns an HTML-styled badge for "Low"
                return f"<span style='background:#d32f2f;color:#fff;border-radius:8px;padding:2px 8px;font-size:0.85em;font-weight:600;margin-left:4px;'>{text}</span>"

            def add_tag(row):
                base = str(row['product_name'])
                if row['quantity'] < row['threshold']:
                    # Render red badge using HTML
                    return (
                        f"{base} {red_badge('Low')}"
                    )
                return base
            display_df['Product (Low)'] = display_df.apply(add_tag, axis=1)
            # Show selected columns, replacing product_name with new one
            cols = ['Product (Low)'] + [c for c in display_df.columns if c not in ['Product (Low)', 'product_name']]
            # Use st.markdown for each row for tag rendering
            st.write("**Low Stock Details:**")
            for _, row in display_df.iterrows():
                prod = row['Product (Low)']
                rest = {k: row[k] for k in display_df.columns if k not in ['Product (Low)', 'product_name']}
                st.markdown(
                    f"- {prod} | Qty: {rest.get('quantity', '')} | Threshold: {rest.get('threshold', '')} | Category: {rest.get('category', '')}",
                    unsafe_allow_html=True
                )
            # Also show styled DataFrame if desired (optional)
            styled_df = low_stock_df.style.apply(highlight_low_stock, axis=1)
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("All items are sufficiently stocked.")

        # Detect low-stock items that haven't been emailed yet
        emailed_products = set()
        if not email_df.empty and 'product_id' in email_df.columns:
            emailed_products = set(email_df['product_id'].dropna().unique())

        low_stock_not_emailed = low_stock_df[~low_stock_df['product_id'].isin(emailed_products)]
        # Deduplicate by product_id and supplier_email to avoid repeat scheduling/emails
        if not low_stock_not_emailed.empty:
            low_stock_not_emailed = low_stock_not_emailed.drop_duplicates(subset=['product_id', 'supplier_email'])

        if not low_stock_not_emailed.empty:
            st.subheader("📧 Sending Restock Alert Emails for Low Stock Items")
            for idx, row in low_stock_not_emailed.iterrows():
                product_id = row['product_id']
                category = row['category']
                quantity = row['quantity']
                threshold = row['threshold']
                price = row['price']

                try:
                    # Enhanced prompt including supplier name and manager contact
                    supplier_name = row.get('supplier_name', '').strip()
                    greeting = f"Dear {supplier_name}," if supplier_name else "Dear Supplier,"
                    prompt = (
                        f"{greeting}\n\n"
                        f"Write a concise and professional restock request email for the following product:\n"
                        f"- Product ID: {product_id}\n"
                        f"- Category: {category}\n"
                        f"- Current Stock: {quantity}\n"
                        f"- Threshold: {threshold}\n\n"
                        f"At the end of the message, include the following contact block:\n"
                        f"Farrukh Khudaykulov\nPC World Store\n+80 70 8566 1551"
                    )
                    response = client.chat.completions.create(
                        model=selected_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=250,
                        temperature=0.6,
                    )
                    email_body = response.choices[0].message.content.strip()
                except Exception as e:
                    st.error(f"OpenAI error for product ID '{product_id}': {e}")
                    log_event(sheets_service, event_type="SUMMARY", product_id=product_id, notes=f"OpenAI error: {e}", status="ERROR", assistant_action="email_prompt")
                    continue

                # Compose and send email via Gmail API
                sender_email = os.getenv("GMAIL_SENDER_EMAIL")
                supplier_email = row.get('supplier_email', '').strip() if 'supplier_email' in row else ''
                subject = f"Urgent Restock Request: {product_id}"

                if not sender_email:
                    st.error("Sender email not configured in environment variables.")
                    continue
                if not supplier_email:
                    st.error(f"Supplier email missing for product ID '{product_id}' in the sheet row.")
                    continue

                # Before logging/sending, check for duplicate in email_df
                duplicate_exists = False
                if not email_df.empty and 'product_id' in email_df.columns and 'supplier_email' in email_df.columns:
                    duplicate_exists = (
                        ((email_df['product_id'] == product_id) & (email_df['supplier_email'] == supplier_email)).any()
                    )
                if duplicate_exists:
                    st.info(f"Already logged/sent email for product ID '{product_id}' and supplier '{supplier_email}'. Skipping.")
                    continue

                message = create_message(sender_email, supplier_email, subject, email_body)

                try:
                    sent_message = send_email_via_gmail(gmail_service, 'me', message)
                    thread_id = sent_message.get('threadId', '')
                    st.success(f"Email sent for product ID '{product_id}' (Thread ID: {thread_id})")

                    # Log the sent email into the "EmailSent" sheet
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Ensure columns: timestamp, product_id, quantity, supplier_email, subject, email_status, response_received, thread_id
                    new_log_entry = [[
                        now_str,
                        row['product_id'],
                        quantity,
                        supplier_email,
                        subject,
                        "SENT",
                        "No",
                        thread_id
                    ]]
                    sheets_service.spreadsheets().values().append(
                        spreadsheetId=SHEET_ID,
                        range=RANGES['email_sent'],
                        valueInputOption='USER_ENTERED',
                        insertDataOption='INSERT_ROWS',
                        body={'values': new_log_entry}
                    ).execute()
                    # Log event to Log sheet
                    log_event(sheets_service, event_type="EMAIL_SENT", product_id=product_id, notes=f"Sent to {supplier_email}", status="SENT", assistant_action="send_email")
                except Exception as e:
                    st.error(f"Failed to send email for product ID '{product_id}': {e}")
                    log_event(sheets_service, event_type="EMAIL_SENT", product_id=product_id, notes=f"Failed to send: {e}", status="ERROR", assistant_action="send_email")
        else:
            st.info("No new low stock items to send emails for.")

        # Merged analytics and summaries from previous tab2
        st.subheader("Overall Metrics")
        total_items = inventory_df['quantity'].sum()
        total_value = (inventory_df['quantity'] * inventory_df['price']).sum()
        st.metric("📦 Total Items in Stock", total_items)
        st.metric("💰 Total Inventory Value", f"${total_value:,.2f}")

        st.subheader("📊 Category Summary")
        category_summary = inventory_df.groupby('category').agg({
            'quantity': 'sum',
            'price': lambda x: (inventory_df.loc[x.index, 'price'] * inventory_df.loc[x.index, 'quantity']).sum()
        }).reset_index()
        category_summary.columns = ['Category', 'Total Quantity', 'Total Value']
        st.dataframe(category_summary)

        # Prepare upgraded prompt for OpenAI summarization, including deeper insight and actionable recommendations
        try:
            shortages_text = ""
            usage_text = ""
            supplier_underdeliver_text = ""
            demand_spike_text = ""
            # Prepare shortages_text from low_stock_df
            if not low_stock_df.empty:
                shortages = []
                for _, row in low_stock_df.iterrows():
                    shortages.append(f"{row['product_name']} (Category: {row['category']}), Quantity: {row['quantity']}, Threshold: {row['threshold']}")
                shortages_text = "\n".join(shortages)
            # Prepare usage_text from usage_df (if available)
            usage_text = ""
            if 'daily_usage' in inventory_df.columns:
                inventory_df['daily_usage'] = pd.to_numeric(inventory_df['daily_usage'], errors='coerce')
                usage_df = inventory_df[inventory_df['daily_usage'] > 0].copy()
                usage_df['days_until_stockout'] = usage_df['quantity'] / usage_df['daily_usage']
                # Fallback for 'name' if not in columns
                if 'name' in usage_df.columns:
                    usage_df['product_name'] = usage_df['name']
                if 'product_name' not in usage_df.columns:
                    usage_df['product_name'] = usage_df['product_id']
                # Only show items with days_until_stockout < 10
                usage_alert = usage_df[usage_df['days_until_stockout'] < 10].copy()
                usage_alert['days_until_stockout'] = usage_alert['days_until_stockout'].round(1)
                usage_rows = []
                for _, row in usage_alert.iterrows():
                    usage_rows.append(
                        f"{row['product_name']} (Qty: {row['quantity']}), Daily Usage: {row['daily_usage']}, Days until stockout: {row['days_until_stockout']}"
                    )
                if usage_rows:
                    usage_text = "\n".join(usage_rows)
            # Prepare supplier underdelivery info (if possible)
            supplier_underdeliver_text = ""
            if not email_df.empty and 'supplier_email' in email_df.columns and 'response_received' in email_df.columns:
                perf = email_df.groupby('supplier_email').agg(
                    emails_sent=('supplier_email', 'count'),
                    replies_received=('response_received', lambda x: (x.str.strip().str.lower() == 'yes').sum())
                ).reset_index()
                underperforming_suppliers = perf[(perf['emails_sent'] > 1) & (perf['replies_received'] < perf['emails_sent'] * 0.7)]
                if not underperforming_suppliers.empty:
                    supplier_underdeliver_text = "\n".join(
                        f"{row['supplier_email']}: {row['replies_received']} replies to {row['emails_sent']} emails"
                        for _, row in underperforming_suppliers.iterrows()
                    )
            # Prepare demand spike info if possible (based on recent increases)
            demand_spike_text = ""
            if 'daily_usage' in inventory_df.columns:
                # Heuristic: spike if daily_usage > 2*median for category
                demand_spikes = []
                for cat, group in inventory_df.groupby('category'):
                    median_usage = group['daily_usage'].median()
                    spikes = group[group['daily_usage'] > 2 * median_usage]
                    for _, row in spikes.iterrows():
                        demand_spikes.append(f"{row['product_name']} in {cat} (Daily Usage: {row['daily_usage']}, Median: {median_usage})")
                if demand_spikes:
                    demand_spike_text = "\n".join(demand_spikes)

            # --- Stock status counts for prompt ---
            stock_status_counts = inventory_df['stock_status'].value_counts().to_dict()
            critical_count = stock_status_counts.get("Critical", 0)
            healthy_count = stock_status_counts.get("Healthy", 0)
            overstocked_count = stock_status_counts.get("Overstocked", 0)
            total_items_count = len(inventory_df)

            stock_status_summary = (
                f"Inventory Stock Status Counts:\n"
                f"- Critical: {critical_count}\n"
                f"- Healthy: {healthy_count}\n"
                f"- Overstocked: {overstocked_count}\n"
                f"- Total items: {total_items_count}\n"
            )

            # --- Stock category explanation for prompt ---
            stock_category_explanation = (
                "Stock Categories Definition:\n"
                "- Critical: quantity < threshold\n"
                "- Healthy: threshold ≤ quantity ≤ threshold × 1.5\n"
                "- Overstocked: quantity > threshold × 1.5\n"
            )

            # Compose upgraded prompt for OpenAI (refactored)
            prompt = (
                "You are an expert inventory management AI assistant. Analyze the following data sections:\n"
                "1. Low stock items (categories, quantities, thresholds):\n"
                f"{shortages_text if shortages_text else '(None)'}\n\n"
                "2. Forecasted stockouts based on usage trends (product, quantity, daily usage, days until stockout):\n"
                f"{usage_text if usage_text else '(None)'}\n\n"
                "3. Suppliers who repeatedly underdeliver (low reply rates):\n"
                f"{supplier_underdeliver_text if supplier_underdeliver_text else '(None)'}\n\n"
                "4. Items showing recent demand spikes (usage much higher than category median):\n"
                f"{demand_spike_text if demand_spike_text else '(None)'}\n\n"
                f"{stock_category_explanation}\n"
                f"{stock_status_summary}\n"
                "Provide a concise inventory summary under 200 words. Include:\n"
                "- Overall stock status (how many items are Critical, Healthy, Overstocked)\n"
                "- Mention items that are low in stock (quantity < threshold) only by name.\n"
                "- Do not include any restock quantities, cost formulas, or calculations.\n"
                "- Focus on clarity and general actionable insight.\n"
                "Use clean formatting and concise wording."
            )
            if shortages_text or usage_text or supplier_underdeliver_text or demand_spike_text:
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.2,
                )
                ai_summary = response.choices[0].message.content.strip()
                # Remove leading "Summary:" (case-insensitive, with or without colon and whitespace)
                import re
                ai_summary = re.sub(r'^\s*summary\s*:\s*', '', ai_summary, flags=re.IGNORECASE)
                # Remove markdown headers like ## or ###
                ai_summary = re.sub(r'^\s*#{1,6}\s*', '', ai_summary, flags=re.MULTILINE)
                # Remove unmatched trailing bold symbols
                ai_summary = re.sub(r'\*\*$', '', ai_summary.strip())
                # Clean trailing hashes or asterisks
                ai_summary = ai_summary.strip().rstrip('#').rstrip('*')
                st.info(f"📌 AI Summary: {ai_summary}")
                st.caption("📌 **Stock Category Rules** — *Critical: quantity < threshold | Healthy: threshold ≤ quantity ≤ 1.5xthreshold | Overstocked: quantity > 1.5xthreshold*")
                log_event(sheets_service, event_type="SUMMARY", product_id=None, notes="Generated enhanced AI inventory summary", status="OK", assistant_action="summarize_inventory")
            else:
                st.info("📌 AI Summary: All categories are within healthy stock levels, no immediate restocking needed.")
        except Exception as e:
            st.info("📌 AI Summary: Most categories are within healthy stock levels, but a few need attention.")
            log_event(sheets_service, event_type="SUMMARY", product_id=None, notes=f"AI summary error: {e}", status="ERROR", assistant_action="summarize_inventory")

        # === Ask a question about inventory (RAG style) ===
        st.subheader("💬 Ask Inventory AI")
        def get_rag_answer(prompt, model):
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                    temperature=0.5,
                    stream=True,
                )
                full_reply = ""
                with st.chat_message("ai"):
                    message_placeholder = st.empty()
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            full_reply += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_reply + "▌")
                    message_placeholder.markdown(full_reply)
                return full_reply
            except Exception as e:
                return f"Error from LLM: {e}"

        user_question = st.chat_input("Ask a question about your inventory (e.g., what's low in Accessories?)")
        if user_question:
            # Turn inventory_df into a text block for grounding (RAG)
            context_rows = inventory_df.fillna("").astype(str).to_dict(orient="records")
            context_text = "\n".join(
                f"ID: {r['product_id']} | Name: {r.get('product_name', '')} | Category: {r['category']} | Quantity: {r['quantity']} | Threshold: {r['threshold']} | Price: {r['price']}"
                for r in context_rows
            )
            rag_prompt = (
                f"You are an inventory analyst AI. Answer the question strictly based on the context below.\n"
                f"Context:\n{context_text}\n\n"
                f"- Only include items where the category field exactly matches any category mentioned in the user's question.\n"
                f"- Do not guess item types based on names or product IDs.\n"
                f"- Use quantity and price to calculate total cost (quantity × price).\n"
                f"- For restock cost, use (threshold - quantity) × price when quantity < threshold.\n"
                f"Question: {user_question}\n"
                f"Answer strictly based on the context. If insufficient information, say so."
            )
            get_rag_answer(rag_prompt, selected_model)

    with tab2:
        st.subheader("📊 Daily Usage-Based Stockout Forecasting")
        # Check if 'daily_usage' column exists
        if 'daily_usage' not in inventory_df.columns:
            st.warning("The 'daily_usage' column is missing from the inventory sheet.")
        else:
            # Ensure daily_usage is numeric
            inventory_df['daily_usage'] = pd.to_numeric(inventory_df['daily_usage'], errors='coerce')
            # Only consider rows with daily_usage > 0
            usage_df = inventory_df[inventory_df['daily_usage'] > 0].copy()
            # Calculate days_until_stockout
            usage_df['days_until_stockout'] = usage_df['quantity'] / usage_df['daily_usage']
            # Select and rename columns for display
            display_cols = ['product_id', 'product_name', 'category', 'quantity', 'daily_usage', 'days_until_stockout']
            # Fallback for 'name' if not in columns
            if 'name' in usage_df.columns:
                usage_df['product_name'] = usage_df['name']
            # If product_name still missing, fallback to product_id
            if 'product_name' not in usage_df.columns:
                usage_df['product_name'] = usage_df['product_id']
            # Rename for display
            usage_df = usage_df.rename(columns={'product_name': 'name'})
            # Keep only required columns
            display_cols = ['product_id', 'name', 'category', 'quantity', 'daily_usage', 'days_until_stockout']
            usage_display = usage_df[display_cols].copy()
            # Format days_until_stockout to 1 decimal
            usage_display['days_until_stockout'] = usage_display['days_until_stockout'].round(1)

            def highlight_stockout(row):
                color = '#ffe6e6' if row['days_until_stockout'] < 7 else ''
                return ['background-color: {}'.format(color) if col == 'days_until_stockout' or col == 'quantity' else '' for col in usage_display.columns]

            st.dataframe(
                usage_display.style.apply(highlight_stockout, axis=1),
                use_container_width=True
            )
            st.caption("Rows highlighted where days until stockout is less than 7 (risk of stockout soon).")

            # -- Insert: Recommend item with lowest days_until_stockout --
            if not usage_display.empty:
                soonest_stockout = usage_display.loc[usage_display['days_until_stockout'].idxmin()]
                st.markdown(
                    f"📌 **Restock Priority Suggestion:**\n"
                    f"The item at highest risk of stockout is **{soonest_stockout['name']}** (ID: {soonest_stockout['product_id']}) "
                    f"with only **{soonest_stockout['days_until_stockout']} days** left based on current usage "
                    f"(Qty: {soonest_stockout['quantity']}, Daily Usage: {soonest_stockout['daily_usage']})."
                )

    with tab3:
        st.subheader("📬 Email Sent Log")
        # ----------- Gmail Reply Tracking -----------
        def fetch_gmail_replies(gmail_service, thread_id, sender_email, supplier_email=None, product_id=None):
            """
            Fetch replies for a given Gmail thread, returns (True, [reply_texts]) if reply is found.
            If thread is missing or no replies, fallback to searching inbox for messages from supplier with matching subject/product_id.
            """
            try:
                thread = gmail_service.users().threads().get(userId='me', id=thread_id, format='full').execute()
                messages = thread.get('messages', [])
                reply_texts = []
                # Assumes first message is sent by us; check for any message not from sender_email
                for msg in messages[1:]:
                    headers = {h['name'].lower(): h['value'] for h in msg.get('payload', {}).get('headers', [])}
                    from_addr = headers.get('from', '')
                    # If the reply is not from us, it's from the supplier
                    if sender_email.lower() not in from_addr.lower():
                        # Try to extract the plain text part of the payload
                        payload = msg.get('payload', {})
                        parts = payload.get('parts', [])
                        body_text = ""
                        if 'body' in payload and payload['body'].get('data'):
                            # Single-part message
                            body_text = base64.urlsafe_b64decode(payload['body']['data']).decode(errors='ignore')
                        else:
                            # Multi-part message
                            for part in parts:
                                if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
                                    body_text = base64.urlsafe_b64decode(part['body']['data']).decode(errors='ignore')
                                    break
                        reply_texts.append(body_text.strip())
                if len(reply_texts) > 0:
                    return (True, reply_texts)
            except Exception as e:
                # Could log error, but don't block UI
                pass
            # Fallback: search inbox for messages from supplier with similar subject/product_id
            if supplier_email and product_id:
                try:
                    # Search for messages from the supplier in inbox with subject containing product_id
                    query = f'in:inbox from:{supplier_email} subject:{product_id}'
                    result = gmail_service.users().messages().list(userId='me', q=query, maxResults=5).execute()
                    messages = result.get('messages', [])
                    reply_texts = []
                    for msg_obj in messages:
                        msg_id = msg_obj.get('id')
                        msg = gmail_service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                        headers = {h['name'].lower(): h['value'] for h in msg.get('payload', {}).get('headers', [])}
                        from_addr = headers.get('from', '')
                        if supplier_email.lower() in from_addr.lower():
                            # Try to extract the plain text part of the payload
                            payload = msg.get('payload', {})
                            parts = payload.get('parts', [])
                            body_text = ""
                            if 'body' in payload and payload['body'].get('data'):
                                body_text = base64.urlsafe_b64decode(payload['body']['data']).decode(errors='ignore')
                            else:
                                for part in parts:
                                    if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
                                        body_text = base64.urlsafe_b64decode(part['body']['data']).decode(errors='ignore')
                                        break
                            if body_text.strip():
                                reply_texts.append(body_text.strip())
                    if reply_texts:
                        return (True, reply_texts)
                except Exception as e:
                    # Could log error, but don't block UI
                    pass
            return (False, [])

        # Only update if email_df has thread_id and response_received columns
        updated_email_df = email_df.copy() if not email_df.empty else pd.DataFrame()
        supplier_replies = {}  # thread_id: [reply_texts]
        if not updated_email_df.empty and 'thread_id' in updated_email_df.columns and 'response_received' in updated_email_df.columns:
            sender_email = os.getenv("GMAIL_SENDER_EMAIL")
            needs_check = updated_email_df['response_received'].str.strip().str.lower() != 'yes'
            updated_flag = False
            for idx, row in updated_email_df[needs_check].iterrows():
                thread_id = row.get('thread_id', '')
                product_id = row.get('product_id', '')
                supplier_email = row.get('supplier_email', '')
                if thread_id:
                    has_reply, reply_texts = fetch_gmail_replies(
                        gmail_service, thread_id, sender_email, supplier_email=supplier_email, product_id=product_id
                    )
                    if has_reply:
                        updated_email_df.at[idx, 'response_received'] = 'Yes'
                        updated_flag = True
                        # Log reply event
                        log_event(sheets_service, event_type="REPLY_RECEIVED", product_id=product_id, notes="Supplier replied", status="RECEIVED", assistant_action="check_reply")
                    if reply_texts:
                        supplier_replies[thread_id] = reply_texts
            # Also collect replies for those already marked as 'Yes'
            already_yes = updated_email_df['response_received'].str.strip().str.lower() == 'yes'
            for idx, row in updated_email_df[already_yes].iterrows():
                thread_id = row.get('thread_id', '')
                supplier_email = row.get('supplier_email', '')
                product_id = row.get('product_id', '')
                if thread_id and thread_id not in supplier_replies:
                    _, reply_texts = fetch_gmail_replies(
                        gmail_service, thread_id, sender_email, supplier_email=supplier_email, product_id=product_id
                    )
                    if reply_texts:
                        supplier_replies[thread_id] = reply_texts
            # If any updates, write back to sheet
            if updated_flag:
                # Write full updated_email_df (including header) back to the EmailSent sheet
                # Prepare values: include header as first row
                values = [list(updated_email_df.columns)] + updated_email_df.astype(str).values.tolist()
                sheets_service.spreadsheets().values().update(
                    spreadsheetId=SHEET_ID,
                    range=RANGES['email_sent'],
                    valueInputOption='USER_ENTERED',
                    body={'values': values}
                ).execute()
            st.dataframe(updated_email_df)
        elif not email_df.empty:
            st.dataframe(email_df)
        else:
            st.info("No email log data available.")

        # ----------- AI Supplier Performance Summary -----------
        st.subheader("📈 Supplier Performance Summary")
        if not updated_email_df.empty and 'supplier_email' in updated_email_df.columns:
            # Add reliability score calculation
            perf = updated_email_df.groupby('supplier_email').agg(
                emails_sent=('supplier_email', 'count'),
                replies_received=('response_received', lambda x: (x.str.strip().str.lower() == 'yes').sum())
            ).reset_index()
            perf['reliability_score'] = perf.apply(
                lambda row: round((row['replies_received'] / row['emails_sent'] * 100) if row['emails_sent'] > 0 else 0, 1),
                axis=1
            )
            st.dataframe(perf)
            st.caption("ℹ️ **Reliability Score** is calculated as: (Replies Received ÷ Emails Sent) × 100")
            # -------- Enhanced AI summary with reply content and reliability score --------
            try:
                # Map supplier_email to all their thread_ids
                supplier_threads = updated_email_df.groupby('supplier_email')['thread_id'].apply(list).to_dict()
                supplier_reply_texts = {}
                for supplier, threads in supplier_threads.items():
                    texts = []
                    for tid in threads:
                        if tid in supplier_replies:
                            texts.extend(supplier_replies[tid])
                    supplier_reply_texts[supplier] = texts

                import re
                urgency_emojis = {
                    "Urgent": "🔴 Urgent",
                    "Delayed": "🟡 Delayed",
                    "Normal": "🟢 Normal"
                }
                ai_enhanced_summaries = []
                for idx, row in perf.iterrows():
                    supplier = row['supplier_email']
                    emails_sent = row['emails_sent']
                    replies_received = row['replies_received']
                    reliability_score = row['reliability_score']
                    reply_texts = supplier_reply_texts.get(supplier, [])
                    reply_excerpt = "\n".join([f"- {txt[:400]}" for txt in reply_texts if txt][:3])  # up to 3 replies, truncated
                    if reply_excerpt.strip() == "":
                        reply_excerpt = "(No reply content available)"
                    # Compose concise prompt for OpenAI supplier performance summary (refactored)
                    prompt = (
                        f"You are an AI assistant evaluating supplier responsiveness.\n"
                        f"Supplier: {supplier}\n"
                        f"Emails sent: {emails_sent}, Replies received: {replies_received}\n"
                        f"Reliability score: {reliability_score}\n"
                        f"Here are sample replies:\n{reply_excerpt}\n\n"
                        f"Summarize the supplier's responsiveness in 2–3 sentences. Begin with the urgency level: Urgency: <Urgent|Delayed|Normal>.\n"
                        f"Evaluate:\n"
                        f"- Whether replies show clear understanding of the restock request\n"
                        f"- Whether any reply indicates confusion or irrelevance (e.g., 'what is this?')\n"
                        f"- Tone and professionalism\n"
                        f"Highlight these factors fairly. Avoid starting any sentence with a comma."
                    )
                    try:
                        summary_resp = client.chat.completions.create(
                            model=selected_model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=300,
                            temperature=0.2,
                        )
                        summary = summary_resp.choices[0].message.content.strip()
                        # Extract urgency flag from the summary
                        urgency_match = re.match(r"Urgency:\s*(Urgent|Delayed|Normal)", summary, re.IGNORECASE)
                        if urgency_match:
                            urgency_flag = urgency_match.group(1).capitalize()
                            summary_content = re.sub(r"^Urgency:\s*(Urgent|Delayed|Normal)[\s\-:]*", "", summary, flags=re.IGNORECASE).strip()
                        else:
                            urgency_flag = "Normal"
                            summary_content = summary
                        # Remove any leading period or whitespace
                        summary_content = summary_content.lstrip(" .")
                        # Remove any redundant "Reliability score: ..." lines from the summary_content (case-insensitive, line start or after a period)
                        summary_content = re.sub(r'(^|\n)[ \t]*Reliability score\s*:\s*.*(\n|$)', '', summary_content, flags=re.IGNORECASE)
                        summary_content = re.sub(r'(^|\n)[ \t]*Reliability\s*:\s*.*(\n|$)', '', summary_content, flags=re.IGNORECASE)
                        emoji_label = urgency_emojis.get(urgency_flag, "🟢 Normal")
                        # Only show the inline badge, not a separate reliability score line after summary
                        supplier_md = (
                            f"**{supplier}** &nbsp; {emoji_label} &nbsp; <span style='background:#e0f7fa;color:#00838f;border-radius:8px;padding:2px 8px;font-size:0.85em;font-weight:600;margin-left:4px;'>Reliability: {reliability_score}%</span>\n\n"
                            f"{summary_content}"
                        )
                        ai_enhanced_summaries.append(supplier_md)
                    except Exception as e:
                        ai_enhanced_summaries.append(f"**{supplier}**:\nUnable to generate summary for this supplier.\n")
                st.markdown("📌 **AI Enhanced Supplier Performance Summary:**", unsafe_allow_html=True)
                for summary_md in ai_enhanced_summaries:
                    st.markdown(summary_md, unsafe_allow_html=True)
            except Exception as e:
                st.info("📌 AI Supplier Performance: Unable to generate enhanced summary.")
        else:
            st.info("No supplier email summary available.")

        # ----------- Calendar Scheduling for Follow-ups -----------
        st.subheader("📅 Schedule Follow-up for Unreplied Emails")
        if not updated_email_df.empty and 'response_received' in updated_email_df.columns:
            unreplied = updated_email_df[updated_email_df['response_received'].str.strip().str.lower() != 'yes']
            # Deduplicate unreplied emails by product_id and supplier_email before scheduling
            if not unreplied.empty:
                unreplied_dedup = unreplied.drop_duplicates(subset=['product_id', 'supplier_email'])
                if st.button("📅 Schedule Calendar Events for Unreplied Emails"):
                    scheduled_count = 0
                    for idx, row in unreplied_dedup.iterrows():
                        subject = row.get('subject', 'Follow-up')
                        supplier_email = row.get('supplier_email', '')
                        product_id = row.get('product_id', '')
                        # Create a calendar event for follow-up
                        try:
                            event = {
                                'summary': f'Follow-up: {subject}',
                                'description': f'Follow up with supplier {supplier_email} for product ID: {product_id}',
                                'start': {
                                    'dateTime': (datetime.now()).strftime('%Y-%m-%dT%H:%M:%S'),
                                    'timeZone': 'UTC',
                                },
                                'end': {
                                    'dateTime': (datetime.now()).strftime('%Y-%m-%dT%H:%M:%S'),
                                    'timeZone': 'UTC',
                                },
                                'attendees': [
                                    {'email': supplier_email}
                                ] if supplier_email else [],
                            }
                            created_event = calendar_service.events().insert(calendarId='primary', body=event).execute()
                            st.success(f"Scheduled follow-up event for supplier '{supplier_email}' (Product: {product_id}, Subject: {subject})")
                            scheduled_count += 1
                            # Log calendar event
                            log_event(sheets_service, event_type="CALENDAR_EVENT", product_id=product_id, notes=f"Follow-up scheduled for {supplier_email}", status="SCHEDULED", assistant_action="calendar_event")
                        except Exception as e:
                            st.error(f"Failed to schedule calendar event for supplier '{supplier_email}' (Product: {product_id}): {e}")
                            log_event(sheets_service, event_type="CALENDAR_EVENT", product_id=product_id, notes=f"Failed: {e}", status="ERROR", assistant_action="calendar_event")
                    if scheduled_count == 0:
                        st.warning("No calendar events scheduled.")
                else:
                    for idx, row in unreplied_dedup.iterrows():
                        subject = row.get('subject', 'Follow-up')
                        supplier_email = row.get('supplier_email', '')
                        product_id = row.get('product_id', '')
                        st.warning(f"Would schedule follow-up for supplier '{supplier_email}' (Product: {product_id}, Subject: {subject})")
            else:
                st.success("All sent emails have received replies. No follow-ups needed.")
        else:
            st.info("No unreplied emails found for follow-up scheduling.")

        st.subheader("📘 Recent AI Log Entries")
        if not log_df.empty:
            st.dataframe(log_df.tail(10))
            # Add download button for log export as CSV
            csv = log_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Log as CSV",
                data=csv,
                file_name="aiims_log.csv",
                mime="text/csv",
            )
        else:
            st.info("Log sheet is currently empty.")


        # 📄 Generate Inventory Performance Report
        st.subheader("📝 Monthly Inventory Report")
        if st.button("📄 Generate Google Docs Report"):
            with st.spinner("Generating report..."):
                try:
                    report_url = generate_monthly_report(inventory_df, email_df, log_df, creds)
                    st.success(f"✅ Report generated: [Open Report]({report_url})")
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")


if __name__ == "__main__":
    main()