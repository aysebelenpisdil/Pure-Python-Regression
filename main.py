import csv
import math
import random


def read_data(filepath):
    x1_list, x2_list, x3_list, y_list = [], [], [], []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x1 = float(row['Present_Tmax'])
                x2 = float(row['LDAPS_Tmax_lapse'])
                x3 = float(row['LDAPS_RHmax'])
                y  = float(row['Next_Tmax'])
                if any(math.isnan(v) for v in (x1, x2, x3, y)):
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            x1_list.append(x1)
            x2_list.append(x2)
            x3_list.append(x3)
            y_list.append(y)
    return x1_list, x2_list, x3_list, y_list


def remove_outliers(x1, x2, x3, y):
    def mean(lst):
        return sum(lst) / len(lst)

    def std(lst, m):
        return math.sqrt(sum((v - m) ** 2 for v in lst) / len(lst))

    m1, m2, m3, my = mean(x1), mean(x2), mean(x3), mean(y)
    s1, s2, s3, sy = std(x1, m1), std(x2, m2), std(x3, m3), std(y, my)

    cx1, cx2, cx3, cy = [], [], [], []
    for i in range(len(y)):
        z1 = abs((x1[i] - m1) / s1) if s1 != 0 else 0
        z2 = abs((x2[i] - m2) / s2) if s2 != 0 else 0
        z3 = abs((x3[i] - m3) / s3) if s3 != 0 else 0
        zy = abs((y[i]  - my) / sy) if sy != 0 else 0
        if z1 <= 3 and z2 <= 3 and z3 <= 3 and zy <= 3:
            cx1.append(x1[i])
            cx2.append(x2[i])
            cx3.append(x3[i])
            cy.append(y[i])
    return cx1, cx2, cx3, cy


def split_data(x1, x2, x3, y, train_ratio=0.70):
    n = len(y)
    indices = list(range(n))
    random.shuffle(indices)
    cut = int(n * train_ratio)
    tr, te = indices[:cut], indices[cut:]
    train = ([x1[i] for i in tr], [x2[i] for i in tr],
             [x3[i] for i in tr], [y[i]  for i in tr])
    test  = ([x1[i] for i in te], [x2[i] for i in te],
             [x3[i] for i in te], [y[i]  for i in te])
    return train, test


def pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num  = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx   = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)))
    dy   = math.sqrt(sum((y[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def linear_regression(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num  = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den  = sum((x[i] - mx) ** 2          for i in range(n))
    b = num / den
    a = my - b * mx
    return a, b


def predict(x, a, b):
    return [a + b * xi for xi in x]


def sse(y_actual, y_pred):
    return sum((y_actual[i] - y_pred[i]) ** 2 for i in range(len(y_actual)))


def save_to_file(predictions, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for val in predictions:
            f.write(f"{val:.6f}\n")


def interactive_prediction(best_name, a, b):
    while True:
        user_input = input(f"{best_name} için bir değer girin (Çıkmak için 'q'): ")
        if user_input.strip().lower() == 'q':
            break
        try:
            x_val = float(user_input)
            y_hat = a + b * x_val
            print(f"Tahmin edilen Next_Tmax: {y_hat:.6f}")
        except ValueError:
            print("Geçersiz giriş. Lütfen sayısal bir değer girin.")


def main():
    x1, x2, x3, y = read_data('Bias_correction_ucl.csv')

    x1, x2, x3, y = remove_outliers(x1, x2, x3, y)

    (tr_x1, tr_x2, tr_x3, tr_y), (te_x1, te_x2, te_x3, te_y) = split_data(x1, x2, x3, y)

    r1 = pearson(tr_x1, tr_y)
    r2 = pearson(tr_x2, tr_y)
    r3 = pearson(tr_x3, tr_y)

    print(f"r1 (Present_Tmax    ile Next_Tmax): {r1:.6f}")
    print(f"r2 (LDAPS_Tmax_lapse ile Next_Tmax): {r2:.6f}")
    print(f"r3 (LDAPS_RHmax     ile Next_Tmax): {r3:.6f}")

    candidates = [
        (abs(r1), tr_x1, te_x1, 'x1 (Present_Tmax)'),
        (abs(r2), tr_x2, te_x2, 'x2 (LDAPS_Tmax_lapse)'),
        (abs(r3), tr_x3, te_x3, 'x3 (LDAPS_RHmax)'),
    ]
    best_abs, best_tr_x, best_te_x, best_name = max(candidates, key=lambda c: c[0])

    print(f"\nEn yüksek mutlak korelasyon: {best_name}  (|r| = {best_abs:.6f})")

    a, b = linear_regression(best_tr_x, tr_y)
    print(f"Model: y_hat = {a:.6f} + {b:.6f} * x")

    tr_preds = predict(best_tr_x, a, b)
    tr_sse   = sse(tr_y, tr_preds)

    print(f"\nEgitim verisi - ilk 10 tahmin:")
    for i in range(min(10, len(tr_preds))):
        print(f"  {i+1:>3}. tahmin: {tr_preds[i]:.6f}  (gercek: {tr_y[i]:.6f})")
    print(f"Egitim SSE: {tr_sse:.6f}")

    save_to_file(tr_preds, 'tahminler_egitim.txt')

    te_preds = predict(best_te_x, a, b)
    te_sse   = sse(te_y, te_preds)

    print(f"\nTest SSE: {te_sse:.6f}")

    save_to_file(te_preds, 'tahminler_test.txt')

    interactive_prediction(best_name, a, b)


if __name__ == '__main__':
    main()
