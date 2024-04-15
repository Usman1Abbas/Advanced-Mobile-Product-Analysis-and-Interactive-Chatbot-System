# Project Overview

This repository contains the implementation of a comprehensive project involving web scraping of mobile products from daraz.pk, data storage, development of an advanced chatbot, and creation of an insightful and interactive dashboard.

## Key Components

### 1. Web Scraping

- **Objective**: Extract mobile product data from daraz.pk, focusing on the first five pages.
- **Tools Used**: Utilized BeautifulSoup for HTML parsing and data extraction.
- **Extracted Information**: Product ID, Name, Price, Company, and Reviews.
- **Data Storage**: Stored product details and reviews in separate CSV files.
- **Key Focus**: Implemented filtering conditions to exclude irrelevant products.

### 2. Database Integration (Bonus)

- **Objective**: Created a structured storage system for the scraped data using MySQL.
- **Database Schema**: Designed tables such as Products (ID, Name, Price, Brand) and Reviews (Review Text, Rating, Product ID).
- **Efficient Querying**: Enabled querying of various data types and performing analyses such as retrieving products within a certain price range or calculating average ratings.

### 3. Chatbot Development

- **Objective**: Developed a chatbot capable of handling user queries based on the scraped data.
- **Natural Language Processing**: Implemented NLP capabilities to understand and respond accurately to user queries.
- **Functionality**: Provided responses to user queries about the best products based on price range, ratings, etc.

### 4. Dashboard Development

- **Objective**: Created an interactive dashboard to visually present the scraped data.
- **Technologies Used**: Utilized Flask for backend development and HTML/CSS for frontend.
- **Dashboard Features**:
  - Input field for querying data (chatbot)
  - Visual representation of total listings, average product price, and ratings
  - Dynamic display of top products based on various criteria
  - Clickable product details with URLs linking back to daraz.pk

## How to Use

1. Clone the repository to your local machine.
2. Set up the required environment and dependencies (Python, Flask, MySQL, etc.).
3. Run the web scraping script to extract data from daraz.pk and store it in CSV files.
4. Set up the MySQL database and import the schema provided in the repository.
5. Run the chatbot and dashboard applications to interact with the scraped data.

## Contributors

- Muhammad Usman

Feel free to contribute to this project by enhancing existing functionalities, adding new features, or improving the documentation. Your contributions are highly appreciated!
