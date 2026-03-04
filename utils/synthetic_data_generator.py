import argparse
import csv
import random
from datetime import datetime, timedelta


def rand_dt(start, end):
    delta = end - start
    sec = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=sec)


def choose_weighted(options, weights):
    r = random.random() * sum(weights)
    acc = 0.0
    for o, w in zip(options, weights):
        acc += w
        if r <= acc:
            return o
    return options[-1]


def gen(rows, out_path):
    now = datetime.utcnow()
    start_date = now - timedelta(days=random.randint(90, 150))
    locations = ["NYC", "Los Angeles", "Chicago", "Houston", "Miami", "San Francisco", "Seattle", "Austin", "Atlanta", "Boston"]
    categories_income = ["rideshare", "delivery", "freelance", "task", "tip"]
    categories_expense = ["groceries", "fuel", "utilities", "fees", "shopping", "dining", "transport", "subscriptions", "healthcare", "mobile_topup"]
    payment_methods = ["Card", "ACH", "Cash", "Wallet", "Bank Transfer"]
    device_types = ["iOS", "Android", "Web"]
    n_users = max(20, min(80, rows // 20))
    users = [f"u{idx+1}" for idx in range(n_users)]
    income_every = {u: random.choice([7, 7, 7, 14, 3]) for u in users}
    income_base = {u: random.uniform(60, 180) for u in users}
    income_jitter = {u: random.uniform(0.05, 0.35) for u in users}
    user_loc = {u: random.choice(locations) for u in users}
    events = []
    for u in users:
        freq = income_every[u]
        t = start_date + timedelta(days=random.randint(0, freq))
        while t < now:
            amt = income_base[u] * (1.0 + random.uniform(-income_jitter[u], income_jitter[u]))
            events.append({
                "user_id": u,
                "transaction_amount": round(max(5.0, amt), 2),
                "transaction_type": "income",
                "timestamp": rand_dt(t, t + timedelta(days=min(freq, 3))).isoformat(),
                "merchant_category": random.choice(categories_income),
                "payment_method": choose_weighted(payment_methods, [4, 3, 1, 2, 2]),
                "location": user_loc[u],
                "device_type": choose_weighted(device_types, [4, 4, 2]),
            })
            t += timedelta(days=freq)
    for u in users:
        day = start_date
        while day < now:
            if random.random() < 0.6:
                k = choose_weighted([1, 2, 3, 4], [5, 4, 2, 1])
                for _ in range(k):
                    cat = random.choice(categories_expense)
                    if cat in ["groceries", "dining", "transport", "mobile_topup"]:
                        amt = random.uniform(3, 40)
                    elif cat in ["fuel", "shopping"]:
                        amt = random.uniform(10, 120)
                    elif cat in ["utilities", "subscriptions", "healthcare"]:
                        amt = random.uniform(15, 200)
                    else:
                        amt = random.uniform(1, 25)
                    ts = rand_dt(day, day + timedelta(days=1))
                    events.append({
                        "user_id": u,
                        "transaction_amount": round(max(1.0, amt), 2),
                        "transaction_type": "expense",
                        "timestamp": ts.isoformat(),
                        "merchant_category": cat,
                        "payment_method": choose_weighted(payment_methods, [5, 2, 1, 2, 2]),
                        "location": user_loc[u],
                        "device_type": choose_weighted(device_types, [5, 4, 2]),
                    })
            day += timedelta(days=1)
    random.shuffle(events)
    events = events[:rows]
    events.sort(key=lambda x: (x["user_id"], x["timestamp"]))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "transaction_amount", "transaction_type", "timestamp", "merchant_category", "payment_method", "location", "device_type"])
        w.writeheader()
        for e in events:
            w.writerow(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_rows", type=int, default=500)
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--out", type=str, default="data/transactions.csv")
    args = parser.parse_args()
    target = random.randint(args.min_rows, args.max_rows)
    gen(target, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
