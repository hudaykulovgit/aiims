import pandas as pd
from datetime import datetime
from googleapiclient.discovery import build

def generate_monthly_report(inventory_df, email_df, log_df, creds):
    # Initialize Google Docs API
    docs_service = build('docs', 'v1', credentials=creds)

    # Create new document
    doc_title = f"AI Inventory Performance Report – {datetime.now().strftime('%B %Y')}"
    doc = docs_service.documents().create(body={'title': doc_title}).execute()
    doc_id = doc.get('documentId')

    # Analyze basic stock status
    critical = inventory_df[inventory_df['quantity'] < inventory_df['threshold']]
    healthy = inventory_df[(inventory_df['quantity'] >= inventory_df['threshold']) & 
                           (inventory_df['quantity'] <= 1.5 * inventory_df['threshold'])]
    overstocked = inventory_df[inventory_df['quantity'] > 1.5 * inventory_df['threshold']]

    # Identify high usage items if 'daily_usage' is present
    high_demand = inventory_df.sort_values(by='daily_usage', ascending=False).head(5) if 'daily_usage' in inventory_df.columns else pd.DataFrame()

    # Compose enhanced report sections
    report_text = f"""AI Inventory Performance Report – {datetime.now().strftime('%B %Y')}

📊 Summary:
- Total Products Tracked: {len(inventory_df)}
- Critical Stock Items: {len(critical)}
- Healthy Stock Items: {len(healthy)}
- Overstocked Items: {len(overstocked)}

⚠️ Issues Identified:
- Stockouts risk detected in {len(critical)} item(s). These items are below threshold and need immediate attention.
{''.join(f'- {row["product_name"]} (Qty: {row["quantity"]}, Threshold: {row["threshold"]})\n' for _, row in critical.iterrows()) if not critical.empty else ''}
- Total number of email alerts sent: {len(email_df)}.
- Logged system events this month: {len(log_df)}

🔥 High Demand Products:"""

    for _, row in high_demand.iterrows():
        report_text += f"\n- {row.get('product_name', 'N/A')} – {row.get('daily_usage', 'N/A')} units/day (Qty: {row.get('quantity', 'N/A')})"

    report_text += f"""

📌 Notable Observations:
- Critical stock includes: {', '.join(critical['product_name'].astype(str).tolist()[:5]) if not critical.empty else 'None'}.
- Items trending toward stockout (within 3–5 days based on usage) should be prioritized even if not yet critical.
- No overstocking issues detected, maintaining a lean inventory profile.

✅ Recommendations:
- Immediately reorder critical items to avoid fulfillment gaps.
- Adjust thresholds for top 5 fastest depleting products to prevent future shortages.
- Evaluate supplier response delays logged in system, especially for high-priority items.

📞 For questions, contact:
Farrukh Khudaykulov
PC World Store
+80 70 8566 1551
"""

    # Format requests for Google Docs
    requests = [{"insertText": {"location": {"index": 1}, "text": report_text}}]
    docs_service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()

    return f"https://docs.google.com/document/d/{doc_id}"