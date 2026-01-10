
from langchain.sql_database import SQLDatabase
import pandas as pd
import os
import sqlite3
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine, text

api_key = ''

df = pd.read_csv("partial_csv.csv", parse_dates=["sale_date"])

# 1. Set your OpenRouter API Key
os.environ["OPENAI_API_KEY"] = "your-openrouter-api-key"  # Replace with your key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# 3. Save DataFrame to SQLite
engine = create_engine("sqlite:///sales.db")
df.to_sql("sales", engine, if_exists="replace", index=False)
db = SQLDatabase(engine)


# 4. Create OpenRouter-compatible LLM using LangChain
llm = ChatOpenAI(
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,     # os.environ["OPENAI_API_KEY"],
    model_name = "mistralai/devstral-small:free"
    # model_name="mistralai/mixtral-8x7b-instruct"
)

# 5. Set up LangChain SQLDatabaseChain
# chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# 6. Natural language question
# question = "What is the total sale amount for 2024?"
# question = "What are total sales?"

# 7. Run the chain and print result
prompt_template = PromptTemplate.from_template("""
You are an expert AI assistant that generates **accurate SQL queries for SQLite** databases based on user 
questions in plain English. 
You are working with the following table:

**Table: sales**

| Column Name             | Description                                     |
|-------------------------|-------------------------------------------------|
| sale_date               | Date of the sale (format: YYYY-MM-DD)          |
| customer_name           | Full name of the customer                       |
| customer_category       | Type of customer (e.g., 'Local', 'Export')      |
| customer_company_size   | Size of the customer's company (e.g., 'Small')  |
| satisfaction_rating     | Rating from 1 to 5                              |
| discount_offered        | 'Yes' or 'No'                                   |
| discount_amount_percent | Discount percentage (integer)                  |
| product_name            | Name of the product                             |
| base_price_per_ton      | Price per ton before discount (integer)        |
| warehouse_name          | Warehouse name                                  |
| warehouse_region        | Region of warehouse (e.g., 'North')             |
| final_tons_sold         | Final tons sold (float)                         |
| sale_amount             | Final sale amount in currency (float)           |

🧠 **Rules for Generating SQL**:
- Use **`customer_category`** when the user mentions customer type like 'Local' or 'Online' or 'International'
- Use **`customer_company_size`** for company size like 'Small', 'Medium', 'Large' or 'Mega'
- Use **`final_tons_sold`** if the user refers to "tons" or "tonnes" sold
- Always filter dates using `sale_date`
- Assume SQLite syntax
- Return **only the SQL query**, no explanations
- For **`customer_name`**, you can assume that the user may sometimes not be fully sure of the name, in that case use the LIKE operator. For example, the user might say that they need sales for customer that is named something like Downtown Grains

Now, generate a SQL query for the following user question:

**User question:** "{question}"
""")


def run_sql(question):

    # ✅ Get SQL query from model
    prompt = prompt_template.format(question=question)
    sql_response = llm.predict(prompt)
    sql_response_text = sql_response

    initial_response = sql_response_text.split("```sql")[0]
    
    if sql_response_text.startswith("```sql"):
        sql_response_text = sql_response_text[6:]  # Remove the first 6 characters
    if sql_response_text.endswith("```"):
        sql_response_text = sql_response_text[:-3]

    sql_response_text = text(sql_response_text)

    print("🧠 Generated SQL:\n", sql_response)

    # ✅ Execute SQL and show result
    try:
        with engine.connect() as conn:
            result = conn.execute(sql_response_text)
            df_result = pd.DataFrame(result.fetchall(), columns=result.keys())
            print("\n\n✅ Query Result:\n", df_result)
            print("\n\n✅ Initial Response:\n", initial_response)
            return df_result
    except Exception as e:
        print("❌ SQL execution failed:", e)
def run():
    keep_running = 1

    while keep_running:

        question = input("What is your query? \n")
        if question not in ["exit", "n", "N"]:
            run_sql(question)
        else:
            exit("Okay. Exiting!")

if __name__ == "__main__":
    print("Welcome to the AI Query Assistant!")
    print("Type your SQL queries in natural language. Type 'exit' to quit.\n")
    print("Example questions:")
    print("- What is the total sale amount for 2024?")
    print("- What are total sales?")
    print("- Show me sales for customer named Downtown Grains")
    print("- How many tons were sold in the North region?")

    run()
