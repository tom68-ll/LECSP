import os
import subprocess
import random
import sqlparse
import tiktoken
spider_train_db_ids = ['company_office', 'storm_record', 'scholar', 'train_station', 'store_product', 'flight_1', 'world_1', 'local_govt_and_lot', 'college_1', 
                       'local_govt_in_alabama', 'insurance_fnol', 'music_1', 'insurance_and_eClaims', 'network_1', 'soccer_1', 'movie_1', 'architecture', 
                       'tracking_grants_for_research', 'race_track', 'entertainment_awards', 'machine_repair', 'behavior_monitoring', 'browser_web', 'voter_1', 
                       'department_management', 'gas_company', 'customers_and_products_contacts', 'tvshow', 'epinions_1', 'local_govt_mdm', 'soccer_2', 
                       'mountain_photos', 'activity_1', 'perpetrator', 'party_people', 'pets_1', 'climbing', 'tracking_orders', 'dorm_1', 'game_1', 'chinook_1', 
                       'film_rank', 'medicine_enzyme_interaction', 'cre_Docs_and_Epenses', 'journal_committee', 'party_host', 'pilot_record', 'singer', 
                       'tracking_software_problems', 'musical', 'student_assessment', 'university_basketball', 'imdb', 'cre_Theme_park', 'performance_attendance', 
                       'school_bus', 'customers_and_addresses', 'geo', 'bike_1', 'allergy_1', 'apartment_rentals', 'e_government', 'culture_company', 'customers_and_invoices', 
                       'cre_Doc_Template_Mgt', 'shop_membership', 'formula_1', 'loan_1', 'protein_institute', 'voter_2', 'aircraft', 'assets_maintenance', 'debate', 
                       'product_catalog', 'workshop_paper', 'inn_1', 'cre_Drama_Workshop_Groups', 'news_report', 'manufactory_1', 'city_record', 'restaurants', 
                       'document_management', 'driving_school', 'college_3', 'company_1', 'customer_deliveries', 'program_share', 'flight_2', 'music_2', 
                       'club_1', 'wta_1', 'museum_visit', 'county_public_safety', 'orchestra', 'real_estate_properties', 'railway', 'csu_1', 'swimming', 
                       'scientist_1', 'wine_1', 'products_gen_characteristics', 'company_employee', 'election_representative', 'school_player', 'concert_singer', 
                       'products_for_hire', 'roller_coaster', 'decoration_competition', 'baseball_1', 'theme_gallery', 'cinema', 'college_2', 'candidate_poll', 
                       'ship_1', 'hospital_1', 'farm', 'network_2', 'academic', 'manufacturer', 'device', 'book_2', 'icfp_1', 'battle_death', 'poker_player', 
                       'sports_competition', 'insurance_policies', 'twitter_1', 'employee_hire_evaluation', 'solvency_ii', 'body_builder', 'cre_Doc_Tracking_DB', 
                       'music_4', 'tracking_share_transactions', 'riding_club', 'customers_campaigns_ecommerce', 'restaurant_1', 'game_injury', 'entrepreneur', 
                       'phone_1', 'small_bank_1', 'yelp', 'flight_company', 'ship_mission', 'student_transcripts_tracking', 'customer_complaints', 'station_weather', 
                       'sakila_1', 'customers_card_transactions', 'student_1', 'department_store', 'wrestler', 'dog_kennels', 'hr_1', 'phone_market', 'match_season', 
                       'store_1', 'course_teach', 'e_learning', 'election', 'wedding', 'coffee_shop', 'car_1', 'school_finance', 'flight_4', 'gymnast', 'cre_Doc_Control_Systems']
spider_dev_db_ids = ['concert_singer', 'pets_1', 'car_1', 'flight_2', 'employee_hire_evaluation', 'cre_Doc_Template_Mgt', 'course_teach', 'museum_visit',
                     'wta_1', 'battle_death', 'student_transcripts_tracking', 'tvshow', 'poker_player', 'voter_1', 'world_1', 'orchestra', 'network_1',
                     'dog_kennels', 'singer', 'real_estate_properties']

db_ids_dataset = {
    "spider-train": spider_train_db_ids,
    "spider-dev": spider_dev_db_ids,
}

CLAUSE_KEYWORDS = ['select', 'from', 'where', 'group by', 'order by', 'limit', 'intersect', 'union', 'except']
JOIN_KEYWORDS = ['join', 'on', 'as']
WHERE_OPS = ['not', 'between', 'in', 'like', 'is', 'exists', '=', '>', '<', '>=', '<=', '!=']
UNIT_OPS = ['-', '+', "*", '/']
AGG_OPS = ['max', 'min', 'count', 'sum', 'avg']
COND_OPS = ['and', 'or']
ORDER_OPS = ['desc', 'asc']
SQL_KEYWORDS = []
SQL_KEYWORDS.extend(CLAUSE_KEYWORDS)
SQL_KEYWORDS.extend(JOIN_KEYWORDS)
SQL_KEYWORDS.extend(WHERE_OPS)
SQL_KEYWORDS.extend(UNIT_OPS)
SQL_KEYWORDS.extend(AGG_OPS)
SQL_KEYWORDS.extend(COND_OPS)
SQL_KEYWORDS.extend(ORDER_OPS)

os.environ["DATA_GYM_CACHE_DIR"] = "tmp/data-gym-cache"
encoding = tiktoken.get_encoding("cl100k_base")
chatgpt_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def get_prompt_length(prompt, model="codex"):
    if model == "codex":
        result = subprocess.run(["node", "codex_prompt_length.mjs", prompt], stdout=subprocess.PIPE)
        prompt_len = eval(result.stdout)
        return prompt_len
    elif model == "chatgpt":
        prompt_len = len(chatgpt_encoding.encode(prompt))
        return prompt_len
    elif model == "gpt3.5":
        raise NotImplementedError


def lexical(query, values):
    if isinstance(query, str):
        for placeholder, value in values.items():
            query = query.replace(placeholder, value)
    elif isinstance(query, list):
        for i in range(len(query)):
            if query[i] in values:
                query[i] = values[query[i]]
    return query


def delexical(query):
    values = {}
    new_query = ""
    in_value = False
    in_col = False
    value = ""
    placeholder_id = 0
    new_query = ""
    for char in query:
        if char == "'":
            in_value = not in_value
            value += char
            if not in_value:
                values[f"value_{placeholder_id}"] = value
                new_query += f"value_{placeholder_id}"
                placeholder_id += 1
                value = ""
        else:
            if not in_value:
                new_query += char
            else:
                value += char
    return new_query, values


def format_query(q, format_type):
    if format_type == 'unnormalized':
        return q["query"]
    elif format_type == 'normalized':
        return q["gold"]["query_normalized"]
    else:
        raise ValueError(f"format_type {format_type} not supported")


def _is_whitespace(sqlparse_token):
    return sqlparse_token.ttype == sqlparse.tokens.Whitespace


def normalize_sql(sql_exp, schema):
    sql_exp = sql_exp.replace('"', "'")
    if sql_exp.count("'") % 2 != 0:  # odd number of single quotes, meaning the value is incomplete or value contains a single quote
        ood_quotes = True
    else:
        ood_quotes = False
    if not ood_quotes:
        sql_exp, values = delexical(sql_exp)
        sql_exp = sql_exp.lower()
    sql_exp = sql_exp.rstrip(";")
    parse = sqlparse.parse(sql_exp)
    sql = parse[0]
    flat_tokens = sql.flatten()
    sql_tokens = [
        token.value for token in flat_tokens if not _is_whitespace(token)
    ]
    sql_lower = ' '.join(sql_tokens)
    sql_lower = sql_lower.replace(' . ', '.')
    for op in AGG_OPS:
        sql_lower = sql_lower.replace(f" {op} (", f" {op}(")
    sql_lower = sql_lower.replace('( ', '(')
    sql_lower = sql_lower.replace(' )', ')')
    sql_lower = sql_lower.replace(' ,', ',')
    sql_lower = sql_lower.rstrip(";")
    sql_lower += ';'
    if not ood_quotes:
        sql_tokens = lexical(sql_tokens, values)
        sql_lower = lexical(sql_lower, values)
    else:
        print("Cannot process the following SQL")
        print(sql_exp, sql_tokens)

    return sql_lower


def petershaw_tokenize_sql(sql_exp):
    """
    Code is adapted from https://github.com/google-research/language/blob/master/language/compgen/nqg/tasks/spider/sql_tokenizer.py"""
    sql_exp = sql_exp.lower()
    sql_exp = sql_exp.rstrip(";")
    parse = sqlparse.parse(sql_exp)
    sql = parse[0]
    flat_tokens = sql.flatten()
    sql_tokens = [
        token.value for token in flat_tokens if not _is_whitespace(token)
    ]
    return sql_tokens


def is_number(token):
    """Check if token is a SQL number literal."""
    # Note that Python's is_numeric() will return False for values like 30.3.
    try:
        float(token)
        return True
    except ValueError:
        return False


petershaw_PLACEHOLDER = "___"


def get_petershaw_template(target):
    """
    Code is adapted from https://github.com/google-research/language/blob/master/language/compgen/nqg/tasks/spider/gen_template_split.py
    Anonymize quoted substrings and numbers in SQL."""
    # First, replace any numeric token.
    tokens = petershaw_tokenize_sql(target)
    template_tokens = []
    for token in tokens:
        if is_number(token):
            template_tokens.append(petershaw_PLACEHOLDER)
        else:
            template_tokens.append(token)
    template = " ".join(template_tokens)

    # Second, replace any subspan surrounded by single or double quotes.
    in_quotes = False
    quote_token = None
    new_template = ""
    for char in template:
        if in_quotes:
            if char == quote_token:
                in_quotes = False
                quote_token = None
        else:
            if char in ("'", "\""):
                in_quotes = True
                quote_token = char
                new_template += petershaw_PLACEHOLDER
            else:
                new_template += char
    return new_template


def find_random_examples(test_q, questions, split="template", deduplicate_demo="nlq"):
    assert split in ["sql", "nlq", "template", None]
    assert deduplicate_demo in ["sql", "nlq", "template"]
    questions_shuffled = random.sample(questions, len(questions))

    seen = set()
    new_questions = []
    for q in questions_shuffled:
        if (split == "nlq" and q["question"] == test_q["question"]) \
                or (split == "sql" and q["query"] == test_q["query"]) \
                or (split == "template" and q["sql_template"] == test_q["sql_template"]):
            continue
        if deduplicate_demo == "nlq" and q["question"] not in seen:
            new_questions.append(q)
            seen.add(q["question"])
        elif deduplicate_demo == "sql" and q["query"] not in seen:
            new_questions.append(q)
            seen.add(q["query"])
        elif deduplicate_demo == "template" and q["sql_template"] not in seen:
            new_questions.append(q)
            seen.add(q["sql_template"])
    return new_questions
