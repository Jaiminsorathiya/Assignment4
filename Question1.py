import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------
# 1) Generate the synthetic dataset (two noisy concentric rings)
# ---------------------------------------------------
def generate_data_per_class(n_samples, r, sigma, label, rng):
    theta = rng.uniform(-np.pi, np.pi, n_samples)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1) * r
    noise = rng.normal(0.0, sigma, size=(n_samples, 2))
    X = circle + noise
    y = np.full(n_samples, label, dtype=int)
    return X, y

def generate_dataset(n_train_per_class=1000,
                     n_test_per_class=10000,
                     r_minus=2.0, r_plus=4.0,
                     sigma=1.0,
                     seed=42):
    rng = np.random.default_rng(seed)

    Xtr_list, ytr_list = [], []
    Xte_list, yte_list = [], []

    for label, r in [(-1, r_minus), (1, r_plus)]:
        Xtr_c, ytr_c = generate_data_per_class(n_train_per_class, r, sigma, label, rng)
        Xte_c, yte_c = generate_data_per_class(n_test_per_class, r, sigma, label, rng)
        Xtr_list.append(Xtr_c); ytr_list.append(ytr_c)
        Xte_list.append(Xte_c); yte_list.append(yte_c)

    X_train = np.vstack(Xtr_list)
    y_train = np.concatenate(ytr_list)
    X_test = np.vstack(Xte_list)
    y_test = np.concatenate(yte_list)

    # shuffle
    rng2 = np.random.default_rng(seed + 1)
    idx_tr = rng2.permutation(len(y_train))
    idx_te = rng2.permutation(len(y_test))

    return X_train[idx_tr], y_train[idx_tr], X_test[idx_te], y_test[idx_te]

X_train, y_train, X_test, y_test = generate_dataset()
print("Train:", X_train.shape, " Test:", X_test.shape)

# ---------------------------------------------------
# 2) CV setup and hyperparameter grids
# ---------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

C_values = [0.1, 1, 10, 100, 1000]
gamma_values = [0.01, 0.1, 1, 10]

hidden_sizes = [5, 10, 20, 40, 60, 80, 100]
alphas = [1e-4, 1e-3, 1e-2]

svm_cv_matrix = np.zeros((len(C_values), len(gamma_values)))
mlp_cv_per_hidden = []

# ---------------------------------------------------
# 3) SVM CV
# ---------------------------------------------------
def cv_score_svm(C, gamma):
    accs = []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        svm = SVC(kernel="rbf", C=C, gamma=gamma)
        svm.fit(X_train[tr_idx], y_train[tr_idx])
        y_val_pred = svm.predict(X_train[val_idx])
        accs.append(accuracy_score(y_train[val_idx], y_val_pred))
    return np.mean(accs)

best_svm_params = None
best_svm_acc = -np.inf

print("\n=== SVM cross-validation ===")
for iC, C in enumerate(C_values):
    for jg, gamma in enumerate(gamma_values):
        mean_acc = cv_score_svm(C, gamma)
        svm_cv_matrix[iC, jg] = mean_acc
        print(f"C={C:>5}, gamma={gamma:>4} -> mean CV accuracy = {mean_acc:.4f}")
        if mean_acc > best_svm_acc:
            best_svm_acc = mean_acc
            best_svm_params = (C, gamma)

C_best, gamma_best = best_svm_params
print(f"\nBest SVM params: C={C_best}, gamma={gamma_best}, "
      f"CV accuracy={best_svm_acc:.4f}, CV error={1-best_svm_acc:.4f}")

svm_final = SVC(kernel="rbf", C=C_best, gamma=gamma_best)
svm_final.fit(X_train, y_train)

y_svm_test = svm_final.predict(X_test)
svm_test_acc = accuracy_score(y_test, y_svm_test)
svm_test_err = 1.0 - svm_test_acc
svm_cm = confusion_matrix(y_test, y_svm_test, labels=[-1, 1])

print("\n=== SVM test performance ===")
print(f"Test accuracy = {svm_test_acc:.4f}")
print(f"Estimated probability of error (0-1 loss) = {svm_test_err:.4f}")
print("Confusion matrix [rows: true (-1,+1), cols: predicted (-1,+1)]:")
print(svm_cm)

# ---------------------------------------------------
# 4) MLP CV
# ---------------------------------------------------
def cv_score_mlp(h, alpha):
    accs = []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        mlp = MLPClassifier(
            hidden_layer_sizes=(h,),
            activation="tanh",
            solver="adam",
            alpha=alpha,
            max_iter=1000,
            random_state=0,
        )
        mlp.fit(X_train[tr_idx], y_train[tr_idx])
        y_val_pred = mlp.predict(X_train[val_idx])
        accs.append(accuracy_score(y_train[val_idx], y_val_pred))
    return np.mean(accs)

best_mlp_params = None
best_mlp_acc = -np.inf

print("\n=== MLP cross-validation ===")
for h in hidden_sizes:
    best_for_h = -np.inf
    for alpha in alphas:
        mean_acc = cv_score_mlp(h, alpha)
        print(f"hidden={h:>3}, alpha={alpha:>6} -> mean CV accuracy = {mean_acc:.4f}")
        if mean_acc > best_for_h:
            best_for_h = mean_acc
        if mean_acc > best_mlp_acc:
            best_mlp_acc = mean_acc
            best_mlp_params = (h, alpha)
    mlp_cv_per_hidden.append(best_for_h)

h_best, alpha_best = best_mlp_params
print(f"\nBest MLP params: hidden={h_best}, alpha={alpha_best}, "
      f"CV accuracy={best_mlp_acc:.4f}, CV error={1-best_mlp_acc:.4f}")

mlp_final = MLPClassifier(
    hidden_layer_sizes=(h_best,),
    activation="tanh",
    solver="adam",
    alpha=alpha_best,
    max_iter=1000,
    random_state=0,
)
mlp_final.fit(X_train, y_train)

y_mlp_test = mlp_final.predict(X_test)
mlp_test_acc = accuracy_score(y_test, y_mlp_test)
mlp_test_err = 1.0 - mlp_test_acc
mlp_cm = confusion_matrix(y_test, y_mlp_test, labels=[-1, 1])

print("\n=== MLP test performance ===")
print(f"Test accuracy = {mlp_test_acc:.4f}")
print(f"Estimated probability of error (0-1 loss) = {mlp_test_err:.4f}")
print("Confusion matrix [rows: true (-1,+1), cols: predicted (-1,+1)]:")
print(mlp_cm)

# ---------------------------------------------------
# 5) FOUR SEPARATE FIGURES (one plot each)
# ---------------------------------------------------

# 5.1 SVM CV heatmap
plt.figure(figsize=(6, 5))
im = plt.imshow(
    svm_cv_matrix,
    origin='lower',
    aspect='auto',
    extent=[min(gamma_values), max(gamma_values),
            min(C_values), max(C_values)],
    cmap='viridis'
)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Gamma')
plt.ylabel('C')
plt.title('SVM Hyperparameter Selection (Accuracy)')
cbar = plt.colorbar(im)
cbar.set_label('CV Mean Accuracy')
plt.xticks(gamma_values)
plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
plt.yticks(C_values)
plt.gca().get_yaxis().set_major_formatter(plt.ScalarFormatter())
plt.tight_layout()
plt.savefig('svm_cv_heatmap.png', dpi=300)
plt.show()

# 5.2 MLP CV curve
plt.figure(figsize=(6, 5))
plt.plot(hidden_sizes, mlp_cv_per_hidden, marker='o')
plt.xlabel('Number of Hidden Neurons')
plt.ylabel('CV Mean Accuracy')
plt.title('MLP Hyperparameter Selection')
plt.grid(True)
plt.tight_layout()
plt.savefig('mlp_cv_curve.png', dpi=300)
plt.show()

# Prepare grid for decision boundaries (shared by both models)
x_min, x_max = X_train[:, 0].min() - 2, X_train[:, 0].max() + 2
y_min, y_max = X_train[:, 1].min() - 2, X_train[:, 1].max() + 2
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)
grid = np.c_[xx.ravel(), yy.ravel()]

# 5.3 SVM decision boundary
plt.figure(figsize=(6, 6))
Z_svm = svm_final.predict(grid).reshape(xx.shape)
plt.contourf(xx, yy, Z_svm, alpha=0.3, levels=[-1, 0, 1], cmap='coolwarm')
plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1],
            s=15, edgecolor='k', facecolor='C0', alpha=0.7, label='class -1')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            s=15, edgecolor='k', facecolor='C1', alpha=0.7, label='class +1')
plt.title(f'SVM Optimal Boundary\nP(Error)={svm_test_err:.4f}')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.gca().set_aspect('equal', 'box')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('svm_decision_boundary.png', dpi=300)
plt.show()

# 5.4 MLP decision boundary
plt.figure(figsize=(6, 6))
Z_mlp = mlp_final.predict(grid).reshape(xx.shape)
plt.contourf(xx, yy, Z_mlp, alpha=0.3, levels=[-1, 0, 1], cmap='coolwarm')
plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1],
            s=15, edgecolor='k', facecolor='C0', alpha=0.7, label='class -1')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            s=15, edgecolor='k', facecolor='C1', alpha=0.7, label='class +1')
plt.title(f'MLP Optimal Boundary\nP(Error)={mlp_test_err:.4f}')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.gca().set_aspect('equal', 'box')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('mlp_decision_boundary.png', dpi=300)
plt.show()
