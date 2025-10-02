def find_best_scale(model_func, scale_factors):
    best_score = -1
    best_factor = None
    best_metrics = None

    for factor in scale_factors:
        print(f"\nEvaluating scale factor: {factor}")
        X_train, X_test, y_train, y_test = model_func(factor)
        acc, prec, rec, f1, spec, auc = train_and_evaluate(X_train, X_test, y_train, y_test)

        print_results(acc, prec, rec, f1, spec, auc)

        if auc > best_score:
            best_score = auc
            best_factor = factor
            best_metrics = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'specificity': spec,
                'auc': auc
            }

    print("\nBest scale factor:", best_factor)
    print("Metrics for best factor:")
    print_results(acc, prec, rec, f1, spec, auc)

    return best_factor, best_metrics