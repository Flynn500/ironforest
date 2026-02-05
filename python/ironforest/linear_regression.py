from ironforest import Array, linalg, column_stack

class LinearRegression:
    """Linear regression model using least squares."""
    
    def __init__(self, fit_intercept: bool = True):
        """
        Initialize linear regression model.
        
        Args:
            fit_intercept: Whether to calculate the intercept for this model.
        """
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self._is_fitted = False
    
    def fit(self, X: Array, y: Array) -> 'LinearRegression':
        """
        Fit linear model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_targets)
        
        Returns:
            self: Fitted estimator
        """
        if self.fit_intercept:
            ones = Array.ones((X.shape[0], 1))
            X_design = column_stack([ones, X])
        else:
            X_design = X

        params, _ = linalg.lstsq(X_design, y)

        if self.fit_intercept:
            intercept_val = params[0]
            self.intercept_ = float(intercept_val) if isinstance(intercept_val, (int, float)) else intercept_val.item()
            self.coef_ = params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params
        
        self._is_fitted = True
        return self
    
    def predict(self, X: Array) -> Array:
        """
        Predict using the linear model.
        
        Args:
            X: Samples of shape (n_samples, n_features)
        
        Returns:
            Predicted values of shape (n_samples,) or (n_samples, n_targets)
        
        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict")
        
        y_pred = X @ self.coef_ # type: ignore
        
        if self.fit_intercept:
            y_pred = y_pred + self.intercept_ 
        
        return y_pred
    
    def score(self, X: Array, y: Array) -> float:
        """
        Return the coefficient of determination (R²) of the prediction.
        
        Args:
            X: Test samples of shape (n_samples, n_features)
            y: True values of shape (n_samples,) or (n_samples, n_targets)
        
        Returns:
            R² score
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling score")
        
        y_pred = self.predict(X)
        
        pred_dif = y - y_pred
        mean_diff = y - y.mean()

        ss_res = (pred_dif * pred_dif).sum()
        ss_tot = (mean_diff * mean_diff).sum()
        
        return 1.0 - (ss_res / ss_tot)
    
    def residuals(self, X: Array, y: Array) -> Array:
        """
        Calculate residuals (y - y_pred).
        
        Args:
            X: Samples of shape (n_samples, n_features)
            y: True values of shape (n_samples,) or (n_samples, n_targets)
        
        Returns:
            Residuals array
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calculating residuals")
        
        return y - self.predict(X)
