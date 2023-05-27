import re

# 3.1 Query Semantics

# SELECT * FROM table_name
# WHERE filter_predicate
# ORACLE LIMIT o
# USING proxy_estimates
# [RECALL | PRECISION] TARGET t
# WITH PROBABILITY p

def query_syntax(query):
    pattern = r"SELECT \* FROM (\w+) WHERE (.+) ORACLE LIMIT (\d+) USING proxy_estimates \[(RECALL|PRECISION)\] TARGET (\d+(\.\d+)?) WITH PROBABILITY (\d+(\.\d+)?)"
    match = re.match(pattern, query)
    if match:
        table_name, filter_predicate, o, proxy_estimates, t, _, p, _ = match.groups()
        return {
            "table_name": table_name,
            "filter_predicate": int(filter_predicate),
            "ORACLE LIMIT": int(o),
            "proxy_estimates": proxy_estimates,
            "target": float(t),
            "delta": float(1- float(p))
        }
    else:
        return "Query Invalid!"

# 3.2 Probabilistic Guarantees
def calculatePR(record, record_accurate):
    record_set = set(record)
    record_accurate_set = set(record_accurate)

    precision = len(record_set.intersection(record_accurate_set)) / len(record_set)
    recall = len(record_set.intersection(record_accurate_set)) / len(record_accurate_set)

    print("Precision =", precision, "Recall =", recall)
    return precision, recall