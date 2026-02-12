"""
Insight generation engine that translates statistical results into
plain-English narratives.

Converts technical analytics into business-friendly explanations.
"""

from typing import List, Dict, Any
import pandas as pd

from core.profiling import DatasetProfile, ColumnProfile
from core.statistics import (
    CorrelationResult,
    RegressionResult,
    TTestResult,
    ANOVAResult,
    RankTestResult,
    ChiSquareResult,
    TwoWayANOVAResult,
    TukeyHSDResult,
    LeveneResult,
    detect_trend,
)


class InsightGenerator:
    """
    Generates human-readable insights from statistical analyses and translates
    technical metrics into action-ready narratives for non-technical users.
    """

    @staticmethod
    def _describe_effect_size(value: float, metric: str = "cohen") -> str:
        """Map numeric effect sizes to qualitative labels."""

        if value is None or pd.isna(value):
            return "not available"

        abs_value = abs(value)
        if metric == "cramer":
            thresholds = (0.1, 0.3, 0.5)
        elif metric == "rank":
            thresholds = (0.1, 0.3, 0.5)
        else:
            thresholds = (0.2, 0.5, 0.8)

        if abs_value < thresholds[0]:
            return "negligible"
        if abs_value < thresholds[1]:
            return "small"
        if abs_value < thresholds[2]:
            return "medium"
        return "large"

    @staticmethod
    def generate_dataset_overview(profile: DatasetProfile) -> List[str]:
        """
        Generate high-level insights about a dataset.
        
        Args:
            profile: DatasetProfile object
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Size and shape
        insights.append(
            f"Dataset contains {profile.row_count:,} rows and "
            f"{profile.column_count} columns, using "
            f"{profile.memory_usage / 1024 / 1024:.1f} MB of memory."
        )
        
        # Data types distribution
        type_summary = ", ".join([
            f"{count} {dtype}" for dtype, count in profile.column_types.items()
        ])
        insights.append(f"Column types: {type_summary}.")
        
        # Data completeness
        completeness_pct = profile.completeness_score * 100
        if completeness_pct >= 95:
            quality = "excellent"
        elif completeness_pct >= 80:
            quality = "good"
        elif completeness_pct >= 60:
            quality = "moderate"
        else:
            quality = "poor"
        
        insights.append(
            f"Data completeness is {quality} at {completeness_pct:.1f}% "
            f"(non-null values)."
        )
        
        # Quality issues
        if profile.quality_issues:
            insights.append(
                f"⚠️ Data quality alerts: {'; '.join(profile.quality_issues)}"
            )
        
        return insights
    
    @staticmethod
    def generate_column_insights(profile: ColumnProfile) -> List[str]:
        """
        Generate insights for a specific column.
        
        Args:
            profile: ColumnProfile object
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Basic info
        insights.append(
            f"'{profile.name}' is a {profile.detected_type} column with "
            f"{profile.unique_count:,} unique values ({profile.unique_percentage:.1f}%)."
        )
        
        # Null handling
        if profile.null_percentage > 0:
            if profile.null_percentage > 20:
                severity = "significant"
            else:
                severity = "minor"
            insights.append(
                f"{severity.capitalize()} missing data: {profile.null_percentage:.1f}% null values."
            )
        
        # Numeric insights
        if profile.numeric_stats:
            stats = profile.numeric_stats
            insights.append(
                f"Range: {stats['min']:.2f} to {stats['max']:.2f}, "
                f"mean: {stats['mean']:.2f}, median: {stats['median']:.2f}."
            )
            
            if abs(stats['skewness']) > 1:
                direction = "right" if stats['skewness'] > 0 else "left"
                insights.append(f"Distribution is heavily skewed {direction}.")
            
            if profile.outliers and len(profile.outliers) > 0:
                insights.append(
                    f"⚠️ Detected {len(profile.outliers)} outliers that may need investigation."
                )
        
        # Categorical insights
        if profile.categorical_stats:
            stats = profile.categorical_stats
            mode_pct = (stats['mode_frequency'] / profile.count) * 100
            insights.append(
                f"Most common value: '{stats['mode']}' ({mode_pct:.1f}% of records)."
            )
        
        # Quality issues
        if profile.quality_issues:
            for issue in profile.quality_issues:
                insights.append(f"⚠️ {issue}")
        
        return insights
    
    @staticmethod
    def generate_correlation_insights(results: List[CorrelationResult]) -> List[str]:
        """
        Generate insights from correlation analysis.
        
        Args:
            results: List of CorrelationResult objects
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Filter significant correlations
        significant = [r for r in results if r.significant and abs(r.correlation) > 0.3]
        
        if not significant:
            insights.append("No significant correlations detected between variables.")
            return insights
        
        # Sort by correlation strength
        significant.sort(key=lambda x: abs(x.correlation), reverse=True)
        
        # Report top correlations
        insights.append(
            f"Found {len(significant)} significant correlation(s) between variables:"
        )
        
        for result in significant[:5]:  # Top 5
            direction = "positive" if result.correlation > 0 else "negative"
            insights.append(
                f"• {result.strength.capitalize()} {direction} correlation between "
                f"'{result.variable1}' and '{result.variable2}' "
                f"(r={result.correlation:.3f}, p={result.p_value:.4f})"
            )
        
        return insights
    
    @staticmethod
    def generate_regression_insights(result: RegressionResult) -> List[str]:
        """
        Generate insights from regression analysis.
        
        Args:
            result: RegressionResult object
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Model fit
        r2_pct = result.r_squared * 100
        insights.append(
            f"Model explains {r2_pct:.1f}% of variance in '{result.dependent_var}' "
            f"(R² = {result.r_squared:.3f}, Adjusted R² = {result.adj_r_squared:.3f})."
        )
        
        # Model quality assessment
        if result.r_squared >= 0.7:
            quality = "strong"
        elif result.r_squared >= 0.4:
            quality = "moderate"
        else:
            quality = "weak"
        insights.append(f"Model fit is {quality}.")
        
        # Significant predictors
        significant_vars = [
            var for var, p in result.p_values.items() if p < 0.05
        ]
        
        if significant_vars:
            insights.append(
                f"Significant predictors: {', '.join(significant_vars)}"
            )
        else:
            insights.append("⚠️ No statistically significant predictors found.")
        
        # Coefficient interpretation
        for var, coef in result.coefficients.items():
            p_val = result.p_values[var]
            if p_val < 0.05:
                direction = "increases" if coef > 0 else "decreases"
                insights.append(
                    f"• One unit increase in '{var}' {direction} '{result.dependent_var}' "
                    f"by {abs(coef):.3f} (p={p_val:.4f})"
                )
        
        return insights
    
    @staticmethod
    def generate_ttest_insights(result: TTestResult) -> List[str]:
        """
        Generate insights from t-test analysis.
        
        Args:
            result: TTestResult object
            
        Returns:
            List of insight strings
        """
        insights = []
        
        test_label = getattr(result, "test_name", "t-test")
        diff = result.group1_mean - result.group2_mean
        direction = result.group1_name if diff > 0 else result.group2_name
        insights.append(
            f"{test_label} comparing '{result.group1_name}' vs '{result.group2_name}' shows "
            f"a mean difference of {abs(diff):.2f} (higher in '{direction}')."
        )

        verdict = "✓" if result.significant else "✗"
        significance_text = "significant" if result.significant else "not significant"
        insights.append(
            f"{verdict} Difference is {significance_text} (t={result.t_statistic:.3f}, p={result.p_value:.4f})."
        )

        descriptor = InsightGenerator._describe_effect_size(result.effect_size)
        size_value = "nan" if pd.isna(result.effect_size) else f"{result.effect_size:.3f}"
        insights.append(f"Effect size is {descriptor} ({size_value}).")
        
        return insights

    @staticmethod
    def generate_rank_test_insights(result: RankTestResult) -> List[str]:
        insights: List[str] = []
        insights.append(
            f"{result.test_name} indicates distributional differences between '{result.group1_name}' and "
            f"'{result.group2_name}'."
        )

        verdict = "✓" if result.significant else "✗"
        desc = "significant" if result.significant else "not significant"
        insights.append(
            f"{verdict} Result is {desc} ({result.statistic_name}={result.statistic_value:.3f}, p={result.p_value:.4f})."
        )

        descriptor = InsightGenerator._describe_effect_size(result.effect_size, metric="rank")
        if descriptor == "not available":
            insights.append("Effect size could not be computed reliably.")
        else:
            insights.append(
                f"Rank-based effect size suggests a {descriptor} shift (value={result.effect_size:.3f})."
            )

        return insights

    @staticmethod
    def generate_chi_square_insights(result: ChiSquareResult) -> List[str]:
        insights: List[str] = []
        verdict = "✓" if result.significant else "✗"
        desc = "associated" if result.significant else "not associated"
        insights.append(
            f"{verdict} Variables '{result.variable1}' and '{result.variable2}' are {desc} "
            f"(χ²={result.chi_square:.3f}, dof={result.dof}, p={result.p_value:.4f})."
        )

        descriptor = InsightGenerator._describe_effect_size(result.cramers_v, metric="cramer")
        if descriptor == "not available":
            insights.append("Unable to compute Cramer's V due to limited data.")
        else:
            insights.append(
                f"Association strength is {descriptor} (Cramer's V={result.cramers_v:.3f})."
            )

        top_categories = result.contingency_table.sum(axis=1).sort_values(ascending=False).head(3)
        if not top_categories.empty:
            readable = ", ".join([f"{idx}: {val}" for idx, val in top_categories.items()])
            insights.append(f"Most represented categories for '{result.variable1}': {readable}.")

        return insights

    @staticmethod
    def generate_two_way_anova_insights(result: TwoWayANOVAResult) -> List[str]:
        insights: List[str] = []
        insights.append(
            f"Two-way ANOVA on '{result.dependent_var}' evaluated factors '{result.factor_a}' and '{result.factor_b}'."
        )

        for label, significant in result.significant_effects.items():
            effect_strength = InsightGenerator._describe_effect_size(result.effect_sizes.get(label))
            verdict = "✓" if significant else "✗"
            desc = "significant" if significant else "not significant"
            insights.append(
                f"{verdict} {label.capitalize()} is {desc} (effect size {effect_strength})."
            )

        top_cells = result.cell_means.stack().sort_values(ascending=False).head(3)
        if not top_cells.empty:
            formatted = ", ".join(
                [f"{idx[0]} / {idx[1]}: {val:.2f}" for idx, val in top_cells.items()]
            )
            insights.append(f"Highest mean combinations → {formatted}.")

        return insights

    @staticmethod
    def generate_tukey_insights(result: TukeyHSDResult) -> List[str]:
        insights: List[str] = []
        if not result.significant_pairs:
            insights.append("Tukey's HSD found no pairwise differences after correction.")
            return insights

        insights.append(
            f"Tukey's HSD identified {len(result.significant_pairs)} significant pair(s) for '{result.group_col}'."
        )

        for idx, (g1, g2) in enumerate(result.significant_pairs[:5], start=1):
            insights.append(f"{idx}. {g1} vs {g2} shows a statistically significant difference.")

        return insights

    @staticmethod
    def generate_levene_insights(result: LeveneResult) -> List[str]:
        insights: List[str] = []
        verdict = "⚠️" if result.significant else "✓"
        desc = "variances differ" if result.significant else "variances appear equal"
        insights.append(
            f"{verdict} Levene's test indicates {desc} across '{result.group_col}' "
            f"(F={result.f_statistic:.3f}, p={result.p_value:.4f})."
        )

        if result.effect_size is not None and not pd.isna(result.effect_size):
            descriptor = InsightGenerator._describe_effect_size(result.effect_size)
            insights.append(
                f"Variance difference effect size is {descriptor} ({result.effect_size:.3f})."
            )

        return insights
    
    @staticmethod
    def generate_anova_insights(result: ANOVAResult) -> List[str]:
        """
        Generate insights from ANOVA analysis.
        
        Args:
            result: ANOVAResult object
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Overall test result
        if result.significant:
            insights.append(
                f"✓ Significant differences detected among {len(result.groups)} groups "
                f"(F={result.f_statistic:.3f}, p={result.p_value:.4f})."
            )
        else:
            insights.append(
                f"✗ No significant differences among groups "
                f"(F={result.f_statistic:.3f}, p={result.p_value:.4f})."
            )
        
        # Group means
        sorted_groups = sorted(result.group_means.items(), key=lambda x: x[1], reverse=True)
        insights.append("Group means (highest to lowest):")
        for group, mean in sorted_groups:
            insights.append(f"  • {group}: {mean:.2f}")
        
        return insights
    
    @staticmethod
    def generate_summary_narrative(
        dataset_profile: DatasetProfile,
        correlations: List[CorrelationResult]
    ) -> str:
        """
        Generate a comprehensive narrative summary of the analysis.
        
        Args:
            dataset_profile: DatasetProfile object
            correlations: List of CorrelationResult objects
            
        Returns:
            Multi-paragraph narrative summary
        """
        paragraphs = []
        
        # Opening
        paragraphs.append(
            f"Analysis of '{dataset_profile.name}' reveals a dataset with "
            f"{dataset_profile.row_count:,} observations across "
            f"{dataset_profile.column_count} variables."
        )
        
        # Data quality
        quality_score = dataset_profile.completeness_score * 100
        if quality_score >= 90:
            quality_statement = "The data quality is excellent with minimal missing values."
        elif quality_score >= 75:
            quality_statement = "The data quality is generally good, though some missing values are present."
        else:
            quality_statement = "Data quality issues should be addressed before drawing strong conclusions."
        paragraphs.append(quality_statement)
        
        # Relationships
        strong_corrs = [c for c in correlations if c.significant and abs(c.correlation) > 0.5]
        if strong_corrs:
            paragraphs.append(
                f"Strong relationships exist between {len(strong_corrs)} variable pairs, "
                f"suggesting potential predictive or explanatory patterns worth investigating."
            )
        
        # Recommendation
        paragraphs.append(
            "Recommend further investigation into the identified patterns and "
            "consideration of additional domain-specific analyses."
        )
        
        return "\n\n".join(paragraphs)
