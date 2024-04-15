from flask import Flask, render_template, request
import random
import pandas as pd
import nltk
from nltk.corpus import stopwords
from fuzzywuzzy import process , fuzz
import re

def filter_dataframe_by_range(df, column, operator, value):
    if operator == 'above' or operator == 'greater' or operator == 'over':
        return df[df[column] > value]
    elif operator == 'below' or operator == 'less':
        return df[df[column] < value]
    elif operator == 'between':
        # Assuming the value is a list [min_value, max_value]
        return df[(df[column] > value[0]) & (df[column] < value[1])]
    else:
        # Handle other cases if needed
        return df

app = Flask(__name__)

# Read the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\DELL\ProgAI\Project\best.csv')
df1 = pd.read_csv(r'C:\Users\DELL\ProgAI\Project\product.csv')

total_listing = len(df1)
avg_price = round(df1['Price'].mean(), 3)
avg_rating = round(df1['Rating'].mean(), 3)
avg_review = round(df1['Total Rating'].mean(), 3)
avg_questions = round(df1['Questions'].mean(), 3)

aa = [f"Total no of Listings   :   {total_listing}", f"Average Product Price   :   {avg_price}", f"Average Product Rating   :   {avg_rating}",
      f"Average Product Reviews   :   {avg_review}", f"Average Questions Asked   :   {avg_questions}"]

@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize variables
    user_query = ""
    chatbot_response = ""

    # Handle user query submission
    if request.method == 'POST':
        user_query = request.form['user_query']
        
        # Your chatbot logic here (replace this with your actual chatbot implementation)
        # For example, you can use the NLTK code you provided
        query = user_query.lower()
        tokens = nltk.word_tokenize(query)
        stop_words = set(stopwords.words('english'))

        # Check for keywords indicating a type of question
        if 'exit' in tokens or 'bye' in tokens:
            chatbot_response = "Goodbye! Have a great day."
        elif 'hi' in tokens or 'hello' in tokens:
            chatbot_response = "Hello! How can I assist you today?"
        # Check for keywords indicating a type of question    

        # Check for both 'price' and 'rating' in tokens
        elif 'price' in tokens and 'rating' in tokens:
            # Find numeric values for both price and rating
            numeric_pattern = re.compile(r'\d+')
            numeric_values = [int(match.group()) for token in tokens if (match := numeric_pattern.search(token))]

            # Find the relevant operators (e.g., above, below)
            operators = ['above', 'below', 'greater', 'less', 'over', 'between']  # Add more if needed
            operator = next((op for op in operators if op in tokens), None)

            if operator:
                # Assuming you have a DataFrame df1 with columns 'Price' and 'Rating'
                price_filtered = filter_dataframe_by_range(df1, 'Price', operator, numeric_values[0])
                rating_filtered = filter_dataframe_by_range(df1, 'Rating', operator, numeric_values[1])

                # Combine the results (you might want to adjust this based on your specific requirement)
                combined_filtered = pd.concat([price_filtered, rating_filtered], axis=0).drop_duplicates()

                chatbot_response = combined_filtered.head().to_string(header=False)
                chatbot_response = '<br><br>'.join(chatbot_response.split('\n'))
            else:
                # Handle case where operator is not found
                chatbot_response = "Please specify a valid range operator (e.g., above, below, between) for both price and rating."


        elif 'range' in tokens or( 'price' in tokens and ( ('below' in tokens or 'lesser' in tokens or 'under' in tokens) and ('above' in tokens or 'greater' in tokens or 'over' in tokens))):

            numeric_pattern = re.compile(r'\d+')
            # Find numeric values in the tokens
            numeric_values = [int(match.group()) for token in tokens if (match := numeric_pattern.search(token))]
            # Find the minimum and maximum values in numeric_values
            min_value = min(numeric_values)
            max_value = max(numeric_values)

            # Filter the DataFrame between min_value and max_value
            filtered = df1[(df1['Price'] >= min_value) & (df1['Price'] <= max_value)]

            chatbot_response = filtered.head().to_string(header = False)
            chatbot_response = '<br><br>'.join(chatbot_response.split('\n'))
            
        elif 'rating' in tokens and (('below' in tokens or 'lesser' in tokens or 'under' in tokens) and ('above' in tokens or 'greater' in tokens or 'over' in tokens)):

            relevant_columns = ['Name', 'Price' , 'Rating' , 'Questions' ,'Brand' , 'Total Rating']
            numeric_pattern = re.compile(r'\d+')
            # Find numeric values in the tokens
            numeric_values = [float(match.group()) for token in tokens if (match := numeric_pattern.search(token))]
            # Find the minimum and maximum values in numeric_values
            min_value = min(numeric_values)
            max_value = max(numeric_values)

            # Filter the DataFrame between min_value and max_value
            filtered = df1[(df1['Rating'] >= min_value) & (df1['Rating'] <= max_value)]

            chatbot_response = filtered[relevant_columns].head().to_string(index=False)
            chatbot_response = '<br><br>'.join(chatbot_response.split('\n'))


        elif 'price' in tokens and ('above' in tokens or 'greater' in tokens or 'over' in tokens):

            numeric_pattern = re.compile(r'\d+')
            # Find numeric values in the tokens
            numeric_values = [int(match.group()) for token in tokens if (match := numeric_pattern.search(token))]

            filtered = df1[df1['Price'] > numeric_values[0]]
            chatbot_response = filtered.head().to_string(header = False)
            chatbot_response = '<br><br>'.join(chatbot_response.split('\n'))
            
        elif 'rating' in tokens and ('above' in tokens or 'greater' in tokens or 'over' in tokens):

            relevant_columns = ['Name', 'Price' , 'Rating' , 'Questions' ,'Brand' , 'Total Rating']

            numeric_pattern = re.compile(r'\d+')
            # Find numeric values in the tokens
            numeric_values = [float(match.group()) for token in tokens if (match := numeric_pattern.search(token))]

            filtered = df1[df1['Rating'] > numeric_values[0]]
            chatbot_response = filtered[relevant_columns].head().to_string(index=False)
            chatbot_response = '<br><br>'.join(chatbot_response.split('\n'))
            
        elif 'rating' in tokens and ('below' in tokens or 'lesser' in tokens or 'under' in tokens):

            relevant_columns = ['Name', 'Price' , 'Rating' , 'Questions' ,'Brand' , 'Total Rating']

            numeric_pattern = re.compile(r'\d+')
            # Find numeric values in the tokens
            numeric_values = [float(match.group()) for token in tokens if (match := numeric_pattern.search(token))]

            filtered = df1[df1['Rating'] < numeric_values[0]]
            chatbot_response = filtered[relevant_columns].head().to_string(index=False)
            chatbot_response = '<br><br>'.join(chatbot_response.split('\n'))

        elif 'stars' in tokens:

            relevant_columns = ['Name', 'Price' , 'Rating' , 'Questions' ,'Brand' , 'Total Rating']

            numeric_pattern = re.compile(r'\d+')
            # Find numeric values in the tokens
            numeric_values = [float(match.group()) for token in tokens if (match := numeric_pattern.search(token))]

            filtered = df1[df1['Rating'] < numeric_values[0]]
            chatbot_response = filtered[relevant_columns].head().to_string(index=False)
            chatbot_response = '<br><br>'.join(chatbot_response.split('\n'))


        elif 'price' in tokens and ('below' in tokens or 'lesser' in tokens or 'under' in tokens):

            numeric_pattern = re.compile(r'\d+')
            # Find numeric values in the tokens
            numeric_values = [int(match.group()) for token in tokens if (match := numeric_pattern.search(token))]

            filtered = df1[df1['Price'] < numeric_values[0]]
            chatbot_response = filtered.head().to_string()
            chatbot_response = '<br><br>'.join(chatbot_response.split('\n'))


        elif 'recommendation' in tokens or 'best' in tokens or 'option' in tokens or 'recomend' in tokens:
            # Find close matches using fuzzy string matching
            matches = process.extract(query, df1['Name'], limit=5)

            # Determine the maximum frequency among the matched products
            max_freq = 0
            matched_products = {}

            for match in matches:
                matched_phone_name = match[0]
                frequency = df1[df1['Name'].apply(lambda x: fuzz.partial_ratio(x, matched_phone_name) >= 80)].shape[0]

                if frequency > max_freq:
                    max_freq = frequency

                matched_products[matched_phone_name] = frequency

            # Display products based on conditions
            if max_freq > 0:
                num_products_to_display = max_freq if any(char.isdigit() for char in query) else 1
                sorted_matches = sorted(matched_products.items(), key=lambda x: x[1], reverse=True)[:num_products_to_display]
                products_to_display = [match[0] for match in sorted_matches]

                # Filter the DataFrame to include the selected products
                filtered = df1[df1['Name'].isin(products_to_display)]

                # Display the results
                chatbot_response = filtered.to_string(header=False)
                chatbot_response = '<br><br>'.join(chatbot_response.split('\n'))
            else:
                chatbot_response = "No suitable recommendations found."


        elif 'top' in tokens or ('rated' in tokens or 'best' in tokens):
            relevant_columns = ['Name', 'Price' , 'Rating' , 'Questions' ,'Brand' , 'Total Rating']
            # Find the numeric value in the tokens (adjust the pattern as needed)
            numeric_pattern = re.compile(r'\d+')
            numeric_values = [int(match.group()) for token in tokens if (match := numeric_pattern.search(token))]

            # Set a default number of top matches (adjust as needed)
            default_top_matches = 5

            # Use the numeric value if available, otherwise use the default
            num_top_matches = numeric_values[0] if numeric_values else default_top_matches

            # Find the best match using fuzzy string matching
            matches = process.extract(query, df1['Name'], limit=num_top_matches)

            # Check if there are matches above a certain threshold (adjust as needed)
            threshold = 80
            top_matches = []

            for match in matches:
                if match[1] >= threshold:
                    matched_phone_name = match[0]
                    matched_row = df1[df1['Name'] == matched_phone_name]
                    top_matches.append(matched_row[relevant_columns])

            if top_matches:
                chatbot_response = f"TOP {num_top_matches} MATCHES WITH HIGHEST RATING ARE:\n"
                for match in top_matches:
                    chatbot_response += match.to_string(header=False) + "\n\n"
            else:
                chatbot_response = "No suitable matches found."



        elif 'compare' in query:
            # Split the query into parts before and after 'and'
            parts = query.split('and', 1)

            if len(parts) == 2:
                part_before, part_after = parts

                # Find the best match for the part before 'and'
                match_before, score_before, idx_before = process.extractOne(part_before.strip(), df1['Name'], scorer=process.fuzz.token_sort_ratio)

                # Find the best match for the part after 'and'
                match_after, score_after, idx_after = process.extractOne(part_after.strip(), df1['Name'], scorer=process.fuzz.token_sort_ratio)

                # Retrieve the entire rows based on the indices
                row_before = df1.iloc[idx_before]
                row_after = df1.iloc[idx_after]

                # Display the results with entire rows
                chatbot_response = (
                    f"\nBEST MATCH FOR PART BEFORE 'and': {match_before} (Score: {score_before})\n"
                    f"{row_before.to_string(index=False)}\n\n"
                    f"\nBEST MATCH FOR PART AFTER 'and': {match_after} (Score: {score_after})\n"
                    f"{row_after.to_string(index=False)}"
                )
            else:
                chatbot_response = "Please include 'and' in your query for comparison."


        else:
            # Find the best match using fuzzy string matching
            matches = process.extractOne(query, df1['Name'])

            # Check if the match is above a certain threshold (adjust as needed)
            threshold = 10
            if matches[1] >= threshold:
                # Get the matched phone name
                matched_phone_name = matches[0]

                # Retrieve the corresponding row from the DataFrame
                matched_row = df1[df1['Name'] == matched_phone_name]

                # Display the result
                chatbot_response = f"CLOSEST MATCH IS  :   {matched_row.to_string(header = False)}"
                chatbot_response = '<br><br>'.join(chatbot_response.split('\n'))

    # Generate another random list for Section 3
    random_data_section3 = aa

    # Pass the user query, chatbot response, random data, DataFrame, and other text to the template
    return render_template('index.html', user_query=user_query, chatbot_response=chatbot_response,
                           df=df, random_data_section3=random_data_section3)

if __name__ == '__main__':
    app.run(debug=True)
