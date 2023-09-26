from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster


def get_elements_by_class_name(driver, class_name):
    elements_list = []
    found_elements = driver.find_elements(By.CLASS_NAME, class_name)
    for element in found_elements:
        elements_list.append(element.text)
    return elements_list


def price_to_int(price):
    digits = []
    for char in price:
        if '0' <= char <= '9':
            digits.append(char)
    
    try:
        integer_price = int(''.join(digits))
    except:
        integer_price = 0
        
    return integer_price


def get_search_list_wb(query='шампунь'):
    chrome_options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=chrome_options)

    url = "https://www.wildberries.ru/catalog/0/search.aspx?search=" + query
    driver.get(url)
    
    time.sleep(5)
    for i in range(10):
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(1)
    
    product_names = get_elements_by_class_name(driver, 'product-card__name')
    product_brands = get_elements_by_class_name(driver, 'product-card__brand')
    product_prices = get_elements_by_class_name(driver, 'price__lower-price')
    
    product_links = driver.find_elements(By.CSS_SELECTOR, 'a.product-card__link')
    product_urls = [link.get_attribute('href') for link in product_links]
    
    driver.quit()
    
    product_titles = [product_brands[i] + ' ' + product_names[i] for i in range(len(product_brands))]
    
    product_prices = list(map(price_to_int, product_prices))
    
    return {'titles': product_titles, 'prices': product_prices, 'links': product_urls}


def get_search_list_ozon(query='шампунь'):
    chrome_options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=chrome_options)
    
    url = 'https://www.ozon.ru/search/?text=' + query + '&from_global=true';
    driver.get(url)
    
    time.sleep(5)
    for i in range(3):
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(1)
    
    css_selector = 'span.c3112-a1.tsHeadline500Medium.c3112-b9'
    price_elements = driver.find_elements(By.CSS_SELECTOR, css_selector)
    product_prices = [element.text for element in price_elements]
    
    product_titles = get_elements_by_class_name(driver, 'tsBody500Medium')[-len(product_prices):]
    product_links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/product/"]')
    product_urls = [link.get_attribute('href') for link in product_links][:2 * len(product_prices):2]
    
    for price in product_prices:
        print(price)
        print(price_to_int(price))
    product_prices = list(map(price_to_int, product_prices))
    
    driver.quit()
    
    return {'titles': product_titles[:36], 'prices': product_prices[:36], 'links': product_urls[:36]}


threshold = 1.0
print('Введите поисковый запрос онлайн-товаров:')
query = input()
print('Производится поиск по запросу.')
wb_result = get_search_list_wb(query)
ozon_result = get_search_list_ozon(query)

all_titles = np.array(ozon_result['titles'] + wb_result['titles'])
all_links = np.array(ozon_result['links'] + wb_result['links'])
all_prices = np.array(ozon_result['prices'] + wb_result['prices'])

vectorizer = TfidfVectorizer(sublinear_tf=True)
tf_idf_features = vectorizer.fit_transform(all_titles).toarray()

linked = linkage(tf_idf_features, method='ward')
cluster_labels = fcluster(linked, 1, criterion='distance')

cluster_ids = pd.Series(cluster_labels).value_counts().index
for cluster_number, cluster_id in enumerate(cluster_ids):
    for i in range(2):
        print('-------------------------------------------------------------------------------')
    print(f'Кластер {cluster_number + 1}:')
    
    indices = np.where(cluster_labels == cluster_id)[0]
    cluster_titles = all_titles[indices]
    cluster_links = all_links[indices]
    cluster_prices = all_prices[indices]
    
    order = np.argsort(cluster_prices)
    for i in order:
        print(cluster_titles[i])
        print(cluster_links[i])
        print(cluster_prices[i], '₽')
        print('-------------------------------------------------------------------------------')