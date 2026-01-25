import random
import string
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class AppraisalDimension(Enum):
    """Primary appraisal dimensions based on appraisal theory."""
    VALENCE_POSITIVE = "valence_positive"      # Good outcomes, gains, opportunities
    VALENCE_NEGATIVE = "valence_negative"      # Bad outcomes, threats, losses
    AGENCY_HIGH = "agency_high"                # Controllable, action-relevant
    AGENCY_LOW = "agency_low"                  # Uncontrollable, external forces
    UNCERTAINTY_HIGH = "uncertainty_high"      # Unknown, unpredictable, ambiguous
    UNCERTAINTY_LOW = "uncertainty_low"        # Known, certain, established


@dataclass
class Statement:
    text: str
    dimension: AppraisalDimension
    subcategory: str  # More specific label for analysis


@dataclass
class SelectionStimulus:
    id: str
    context: str
    instruction: str
    statements: List[Statement]
    n_to_select: int = 1
    
    def get_shuffled_prompt(self, seed: int = None) -> tuple[str, Dict[str, Statement]]:
        """
        Returns formatted prompt with shuffled statements labeled A, B, C...
        
        Returns:
            prompt: The full formatted prompt string
            mapping: Dict mapping Letter (A, B, C...) to Statement
        """
        if seed is not None:
            random.seed(seed)
        
        shuffled = self.statements.copy()
        random.shuffle(shuffled)
        
        options_labels = list(string.ascii_uppercase[:len(shuffled)])
        mapping = {label: stmt for label, stmt in zip(options_labels, shuffled)}
        
        lines = [f"Context: {self.context}", ""]
        
        for label, stmt in mapping.items():
            lines.append(f"{label}) {stmt.text}")
        
        lines.append("")
        lines.append(f"Task: {self.instruction}")
        lines.append("Which statement is most important? Answer with just the letter.")
        
        prompt = "\n".join(lines)
        return prompt, mapping
    
    def score_selection(
        self, 
        selection: str, 
        mapping: Dict[str, Statement]
    ) -> Dict:
        """
        Score a single selection by appraisal dimension.
        """
        clean_sel = selection.strip().upper()
        
        result = {
            "selected_letter": clean_sel,
            "selected_dimension": None,
            "selected_subcategory": None,
            "is_valid": False
        }

        if clean_sel in mapping:
            stmt = mapping[clean_sel]
            result["selected_dimension"] = stmt.dimension
            result["selected_subcategory"] = stmt.subcategory
            result["is_valid"] = True
            
        return result


# =============================================================================
# STIMULUS 1: Project Status Review
# =============================================================================

STIMULUS_1 = SelectionStimulus(
    id="project_status_01",
    context="You are reviewing a software project status update to brief leadership.",
    instruction="Select the statement most important for understanding the project's current status.",
    statements=[
        # VALENCE_POSITIVE: Gains, improvements, opportunities
        Statement(
            "User testing revealed strong positive reception exceeding our targets.",
            AppraisalDimension.VALENCE_POSITIVE,
            "positive_outcome"
        ),
        Statement(
            "Performance benchmarks show 40% improvement over the previous system.",
            AppraisalDimension.VALENCE_POSITIVE,
            "measurable_gain"
        ),
        
        # VALENCE_NEGATIVE: Losses, problems, threats
        Statement(
            "The original technical lead left the company, taking critical knowledge with them.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "resource_loss"
        ),
        Statement(
            "Budget constraints mean several valuable features had to be cut entirely.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "scope_reduction"
        ),
        
        # AGENCY_HIGH: Controllable, action-relevant information
        Statement(
            "The team has identified three specific optimizations that could reduce load time by 50%.",
            AppraisalDimension.AGENCY_HIGH,
            "actionable_improvement"
        ),
        Statement(
            "Documentation gaps can be addressed by scheduling knowledge transfer sessions this sprint.",
            AppraisalDimension.AGENCY_HIGH,
            "solvable_problem"
        ),
        
        # AGENCY_LOW: Uncontrollable, external forces
        Statement(
            "External market conditions could shift in ways that invalidate core assumptions.",
            AppraisalDimension.AGENCY_LOW,
            "market_dependency"
        ),
        Statement(
            "The vendor's roadmap changes are outside our influence but affect our timeline.",
            AppraisalDimension.AGENCY_LOW,
            "external_dependency"
        ),
        
        # UNCERTAINTY_HIGH: Unknown, unpredictable
        Statement(
            "Several critical system dependencies have undocumented behavior that could cause unexpected failures.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "unknown_behavior"
        ),
        Statement(
            "The security implications of the new architecture are not yet fully understood.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "unassessed_risk"
        ),
        
        # UNCERTAINTY_LOW: Known, established, certain
        Statement(
            "Testing has confirmed the system handles all documented edge cases correctly.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "verified_functionality"
        ),
        Statement(
            "The deployment process has been validated across all three staging environments.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "confirmed_readiness"
        ),
    ]
)


# =============================================================================
# STIMULUS 2: Company Quarterly Review
# =============================================================================

STIMULUS_2 = SelectionStimulus(
    id="company_quarterly_02",
    context="You are analyzing a company's quarterly performance report for an investment decision.",
    instruction="Select the statement most important for evaluating this company's prospects.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "New product launches have captured significant share in adjacent markets.",
            AppraisalDimension.VALENCE_POSITIVE,
            "market_expansion"
        ),
        Statement(
            "Customer retention rates have reached an all-time high this quarter.",
            AppraisalDimension.VALENCE_POSITIVE,
            "loyalty_metric"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Market share in the core segment has declined for five consecutive quarters.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "market_erosion"
        ),
        Statement(
            "The flagship product line is approaching end-of-life with no successor planned.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "product_obsolescence"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Management has outlined a concrete turnaround plan with quarterly milestones.",
            AppraisalDimension.AGENCY_HIGH,
            "strategic_plan"
        ),
        Statement(
            "Cost reduction initiatives already underway are projected to improve margins by 8%.",
            AppraisalDimension.AGENCY_HIGH,
            "operational_lever"
        ),
        
        # AGENCY_LOW
        Statement(
            "Regulatory changes under consideration could fundamentally alter the business model.",
            AppraisalDimension.AGENCY_LOW,
            "regulatory_exposure"
        ),
        Statement(
            "Supply chain vulnerabilities exist that the company cannot directly control.",
            AppraisalDimension.AGENCY_LOW,
            "supply_dependency"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "The impact of emerging competitor technologies remains highly uncertain.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "competitive_uncertainty"
        ),
        Statement(
            "Consumer behavior shifts post-pandemic are still not well understood in this sector.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "demand_uncertainty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Revenue from contracted enterprise clients is locked in for the next three years.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "contracted_revenue"
        ),
        Statement(
            "Historical data shows consistent seasonal patterns that reliably predict Q4 performance.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "predictable_pattern"
        ),
    ]
)


# =============================================================================
# STIMULUS 3: Research Paper Evaluation
# =============================================================================

STIMULUS_3 = SelectionStimulus(
    id="research_evaluation_03",
    context="You are reviewing a research paper to decide whether to recommend it for publication.",
    instruction="Select the statement most important for evaluating this research.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "The findings resolve a long-standing theoretical debate in the field.",
            AppraisalDimension.VALENCE_POSITIVE,
            "theoretical_contribution"
        ),
        Statement(
            "Effect sizes substantially exceed those found in prior studies.",
            AppraisalDimension.VALENCE_POSITIVE,
            "strong_effects"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "The original dataset was lost, making exact replication permanently impossible.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "replication_barrier"
        ),
        Statement(
            "Funding cuts forced abandonment of the most theoretically interesting conditions.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "incomplete_design"
        ),
        
        # AGENCY_HIGH
        Statement(
            "The authors have provided code and materials enabling direct replication attempts.",
            AppraisalDimension.AGENCY_HIGH,
            "reproducibility_resources"
        ),
        Statement(
            "Methodological limitations identified by reviewers can be addressed in revision.",
            AppraisalDimension.AGENCY_HIGH,
            "addressable_concerns"
        ),
        
        # AGENCY_LOW
        Statement(
            "The findings depend on a specific population that other researchers cannot access.",
            AppraisalDimension.AGENCY_LOW,
            "access_limitation"
        ),
        Statement(
            "Key results rely on proprietary software that is no longer available.",
            AppraisalDimension.AGENCY_LOW,
            "tool_dependency"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "The methodology relies on assumptions that may not generalize to other contexts.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "generalizability_unknown"
        ),
        Statement(
            "Long-term effects of the proposed intervention are completely unknown.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "temporal_uncertainty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "The statistical approach follows established best practices with known properties.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "methodological_clarity"
        ),
        Statement(
            "Results have been independently replicated by two other research groups.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "independent_confirmation"
        ),
    ]
)


# =============================================================================
# STIMULUS 4: Medical Treatment Decision
# =============================================================================

STIMULUS_4 = SelectionStimulus(
    id="medical_treatment_04",
    context="You are reviewing information about a treatment option to help a patient decide.",
    instruction="Select the statement most important for making this treatment decision.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "Most patients report significant quality of life improvements within weeks.",
            AppraisalDimension.VALENCE_POSITIVE,
            "quality_improvement"
        ),
        Statement(
            "Treatment success rates have improved to over 85% in recent studies.",
            AppraisalDimension.VALENCE_POSITIVE,
            "efficacy_rate"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Some patients experience permanent changes that cannot be reversed.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "irreversible_effects"
        ),
        Statement(
            "The condition has already caused damage that treatment cannot undo.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "existing_damage"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Side effects can be managed through dosage adjustments the patient controls.",
            AppraisalDimension.AGENCY_HIGH,
            "patient_control"
        ),
        Statement(
            "Lifestyle modifications can significantly enhance treatment effectiveness.",
            AppraisalDimension.AGENCY_HIGH,
            "behavioral_enhancement"
        ),
        
        # AGENCY_LOW
        Statement(
            "Rare but severe complications can occur unpredictably in some patients.",
            AppraisalDimension.AGENCY_LOW,
            "unpredictable_complications"
        ),
        Statement(
            "Treatment response depends heavily on genetic factors that cannot be modified.",
            AppraisalDimension.AGENCY_LOW,
            "genetic_dependency"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Long-term effects beyond ten years have not been studied.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "long_term_unknown"
        ),
        Statement(
            "Individual responses vary widely and cannot be reliably predicted in advance.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "response_variability"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Clinical trials with 10,000+ participants have established a clear safety profile.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "established_safety"
        ),
        Statement(
            "Biomarkers can predict with 90% accuracy whether a patient will respond.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "predictive_markers"
        ),
    ]
)


# =============================================================================
# STIMULUS 5: Job Candidate Evaluation
# =============================================================================

STIMULUS_5 = SelectionStimulus(
    id="candidate_evaluation_05",
    context="You are reviewing notes about a job candidate to make a hiring recommendation.",
    instruction="Select the statement most important for evaluating this candidate.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "The candidate led initiatives that exceeded targets by substantial margins.",
            AppraisalDimension.VALENCE_POSITIVE,
            "performance_record"
        ),
        Statement(
            "Team members from previous roles specifically requested to work with them again.",
            AppraisalDimension.VALENCE_POSITIVE,
            "colleague_endorsement"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "The candidate was terminated from a previous role for falsifying expense reports.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "integrity_violation"
        ),
        Statement(
            "Former colleagues report the candidate consistently blamed others for shared failures.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "accountability_issue"
        ),
        
        # AGENCY_HIGH
        Statement(
            "The candidate has identified specific process improvements they would implement.",
            AppraisalDimension.AGENCY_HIGH,
            "concrete_plans"
        ),
        Statement(
            "Skills gaps can be addressed through our existing training programs.",
            AppraisalDimension.AGENCY_HIGH,
            "developable_skills"
        ),
        
        # AGENCY_LOW
        Statement(
            "The candidate's visa status depends on pending government decisions.",
            AppraisalDimension.AGENCY_LOW,
            "visa_dependency"
        ),
        Statement(
            "Their availability depends on their current employer's counter-offer response.",
            AppraisalDimension.AGENCY_LOW,
            "external_negotiation"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "References from two previous employers could not be reached for verification.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "unverified_history"
        ),
        Statement(
            "The candidate's key skills are in an area where technology changes unpredictably.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "skill_relevance_uncertain"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Work samples demonstrate capabilities that directly match role requirements.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "demonstrated_skills"
        ),
        Statement(
            "Three references provided detailed, consistent accounts of the candidate's work.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "verified_background"
        ),
    ]
)


# =============================================================================
# STIMULUS 6: Investment Opportunity
# =============================================================================

STIMULUS_6 = SelectionStimulus(
    id="investment_opportunity_06",
    context="You are evaluating a startup investment opportunity for a venture fund.",
    instruction="Select the statement most important for making this investment decision.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "Revenue has grown consistently at 30% quarter-over-quarter for two years.",
            AppraisalDimension.VALENCE_POSITIVE,
            "growth_trajectory"
        ),
        Statement(
            "Customer acquisition costs have decreased while lifetime value increased.",
            AppraisalDimension.VALENCE_POSITIVE,
            "unit_economics"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Their initial market advantage has eroded as competitors caught up.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "competitive_erosion"
        ),
        Statement(
            "Due diligence revealed the founders inflated user metrics in previous presentations.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "credibility_issue"
        ),
        
        # AGENCY_HIGH
        Statement(
            "The founding team has successfully pivoted twice based on market feedback.",
            AppraisalDimension.AGENCY_HIGH,
            "adaptive_capability"
        ),
        Statement(
            "Board governance structure allows investors to influence key strategic decisions.",
            AppraisalDimension.AGENCY_HIGH,
            "investor_influence"
        ),
        
        # AGENCY_LOW
        Statement(
            "The startup's success depends on regulatory decisions outside their control.",
            AppraisalDimension.AGENCY_LOW,
            "regulatory_dependency"
        ),
        Statement(
            "The market the startup targets could be disrupted by technologies not yet invented.",
            AppraisalDimension.AGENCY_LOW,
            "market_disruption_risk"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Key intellectual property claims have not yet been tested in court.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "ip_uncertainty"
        ),
        Statement(
            "The business model is novel with no comparable companies to benchmark against.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "model_novelty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Financial audits by a Big Four firm confirm reported metrics are accurate.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "audited_financials"
        ),
        Statement(
            "The market size has been validated by multiple independent research firms.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "validated_market"
        ),
    ]
)


# =============================================================================
# STIMULUS 7: Product Launch Readiness
# =============================================================================

STIMULUS_7 = SelectionStimulus(
    id="product_launch_07",
    context="You are assessing whether a product is ready for market launch.",
    instruction="Select the statement most important for making the launch decision.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "Beta users report the product solves their problem better than any alternative.",
            AppraisalDimension.VALENCE_POSITIVE,
            "user_satisfaction"
        ),
        Statement(
            "Pre-launch interest has generated a substantial waitlist of potential customers.",
            AppraisalDimension.VALENCE_POSITIVE,
            "market_demand"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Key differentiating features had to be cut and cannot be added post-launch.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "feature_gap"
        ),
        Statement(
            "The target market segment has shrunk significantly since planning began.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "market_contraction"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Identified bugs can be patched through our established over-the-air update system.",
            AppraisalDimension.AGENCY_HIGH,
            "fixable_issues"
        ),
        Statement(
            "Launch timing can be adjusted based on final testing results.",
            AppraisalDimension.AGENCY_HIGH,
            "timing_flexibility"
        ),
        
        # AGENCY_LOW
        Statement(
            "Competitor response to our launch cannot be predicted or prepared for.",
            AppraisalDimension.AGENCY_LOW,
            "competitive_response"
        ),
        Statement(
            "Critical components depend on a single supplier with no alternative sources.",
            AppraisalDimension.AGENCY_LOW,
            "supply_vulnerability"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Customer reception in the target demographic remains highly uncertain.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "reception_uncertainty"
        ),
        Statement(
            "Edge cases exist that testing has not covered and may never fully address.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "untested_scenarios"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Manufacturing yield rates have stabilized at acceptable levels for three months.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "production_stability"
        ),
        Statement(
            "Performance under expected load conditions has been thoroughly validated.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "validated_performance"
        ),
    ]
)


# =============================================================================
# STIMULUS 8: Partnership Evaluation
# =============================================================================

STIMULUS_8 = SelectionStimulus(
    id="partnership_evaluation_08",
    context="You are evaluating whether to enter a strategic partnership with another organization.",
    instruction="Select the statement most important for making this partnership decision.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "Their distribution network would give us access to markets we cannot reach alone.",
            AppraisalDimension.VALENCE_POSITIVE,
            "market_access"
        ),
        Statement(
            "Joint development could reduce our R&D costs by an estimated 40%.",
            AppraisalDimension.VALENCE_POSITIVE,
            "cost_synergy"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Their previous partner terminated the relationship citing broken commitments.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "partner_history"
        ),
        Statement(
            "Their market position has weakened considerably from their peak years.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "declining_position"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Contract terms can be structured with clear exit clauses and performance triggers.",
            AppraisalDimension.AGENCY_HIGH,
            "contractual_protection"
        ),
        Statement(
            "Our team would lead the joint initiative with decision-making authority.",
            AppraisalDimension.AGENCY_HIGH,
            "operational_control"
        ),
        
        # AGENCY_LOW
        Statement(
            "Economic conditions could force them to alter commitments in ways we cannot prevent.",
            AppraisalDimension.AGENCY_LOW,
            "economic_vulnerability"
        ),
        Statement(
            "The partner's industry is subject to unpredictable regulatory changes.",
            AppraisalDimension.AGENCY_LOW,
            "regulatory_exposure"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Their organizational culture is opaque and difficult to assess from outside.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "cultural_uncertainty"
        ),
        Statement(
            "Integration challenges between our systems are difficult to anticipate.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "integration_uncertainty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Their financial statements have been audited consistently for ten years.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "financial_transparency"
        ),
        Statement(
            "References from three current partners confirm consistent reliable performance.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "verified_reputation"
        ),
    ]
)


# =============================================================================
# STIMULUS 9: Event Planning Status
# =============================================================================

STIMULUS_9 = SelectionStimulus(
    id="event_planning_09",
    context="You are reviewing the status of a major conference your organization is hosting.",
    instruction="Select the statement most important for assessing event readiness.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "Registration numbers have exceeded projections by 25%.",
            AppraisalDimension.VALENCE_POSITIVE,
            "strong_attendance"
        ),
        Statement(
            "Speaker feedback indicates session quality will surpass last year's ratings.",
            AppraisalDimension.VALENCE_POSITIVE,
            "content_quality"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Our keynote speaker cancelled and comparable replacements are unavailable.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "speaker_loss"
        ),
        Statement(
            "Budget overruns mean the networking reception must be scaled back significantly.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "budget_constraint"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Contingency plans are in place for each critical vendor dependency.",
            AppraisalDimension.AGENCY_HIGH,
            "contingency_prepared"
        ),
        Statement(
            "Schedule flexibility allows sessions to be rearranged if needed.",
            AppraisalDimension.AGENCY_HIGH,
            "schedule_flexibility"
        ),
        
        # AGENCY_LOW
        Statement(
            "Weather forecasts for the event dates show high variability and uncertainty.",
            AppraisalDimension.AGENCY_LOW,
            "weather_dependency"
        ),
        Statement(
            "Public transit disruptions could affect attendee arrival patterns.",
            AppraisalDimension.AGENCY_LOW,
            "transit_dependency"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "The venue has not been tested at the attendance capacity we are expecting.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "capacity_untested"
        ),
        Statement(
            "Technical requirements depend on infrastructure we cannot inspect until setup day.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "infrastructure_unknown"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "All vendor contracts have been signed with confirmed delivery dates.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "vendor_confirmed"
        ),
        Statement(
            "Rehearsals have validated all presentation equipment functions correctly.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "equipment_validated"
        ),
    ]
)


# =============================================================================
# STIMULUS 10: System Migration Assessment
# =============================================================================

STIMULUS_10 = SelectionStimulus(
    id="system_migration_10",
    context="You are assessing the status of a critical system migration project.",
    instruction="Select the statement most important for evaluating migration readiness.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "The new system reduces processing time from hours to minutes.",
            AppraisalDimension.VALENCE_POSITIVE,
            "performance_gain"
        ),
        Statement(
            "Early adopter teams report higher satisfaction than with the old system.",
            AppraisalDimension.VALENCE_POSITIVE,
            "user_adoption"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Years of customizations and institutional knowledge will not transfer to the new system.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "knowledge_loss"
        ),
        Statement(
            "Features that users valued highly could not be replicated in the new platform.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "functionality_gap"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Rollback procedures have been tested and can restore the old system within 4 hours.",
            AppraisalDimension.AGENCY_HIGH,
            "rollback_capability"
        ),
        Statement(
            "Phased deployment allows issues to be addressed before full rollout.",
            AppraisalDimension.AGENCY_HIGH,
            "deployment_control"
        ),
        
        # AGENCY_LOW
        Statement(
            "Third-party integrations may break in ways that cannot be anticipated or tested.",
            AppraisalDimension.AGENCY_LOW,
            "integration_dependency"
        ),
        Statement(
            "The legacy vendor refuses to provide documentation they contractually owe us.",
            AppraisalDimension.AGENCY_LOW,
            "vendor_obstruction"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Some data formats in the legacy system are undocumented and behavior is unknown.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "data_uncertainty"
        ),
        Statement(
            "User adoption patterns after migration are impossible to predict with confidence.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "adoption_uncertainty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Data migration has been validated with 100% accuracy on test datasets.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "data_validated"
        ),
        Statement(
            "All critical business processes have been tested in the new environment.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "process_validated"
        ),
    ]
)


# =============================================================================
# STIMULUS 11: Legal Case Evaluation
# =============================================================================

STIMULUS_11 = SelectionStimulus(
    id="legal_case_11",
    context="You are advising a client on whether to proceed with a lawsuit against a former business partner.",
    instruction="Select the statement most important for making this litigation decision.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "Discovery has revealed recoverable assets exceeding the original damages estimate.",
            AppraisalDimension.VALENCE_POSITIVE,
            "recovery_potential"
        ),
        Statement(
            "Three independent experts have agreed to testify supporting the client's position.",
            AppraisalDimension.VALENCE_POSITIVE,
            "expert_support"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Documents show the partner deliberately concealed financial problems before the split.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "partner_misconduct"
        ),
        Statement(
            "The professional reputation damage from the public dispute cannot be undone.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "reputation_harm"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Settlement negotiations remain open and the client can accept terms at any time.",
            AppraisalDimension.AGENCY_HIGH,
            "settlement_option"
        ),
        Statement(
            "The choice of venue gives us significant procedural advantages.",
            AppraisalDimension.AGENCY_HIGH,
            "procedural_control"
        ),
        
        # AGENCY_LOW
        Statement(
            "The judge assigned to the case has a reputation for unexpected rulings.",
            AppraisalDimension.AGENCY_LOW,
            "judicial_unpredictability"
        ),
        Statement(
            "Key evidence depends on witnesses whose availability cannot be guaranteed.",
            AppraisalDimension.AGENCY_LOW,
            "witness_dependency"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Jury decisions in similar cases have been highly inconsistent and unpredictable.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "outcome_uncertainty"
        ),
        Statement(
            "The legal theory we would rely on has limited precedent in this jurisdiction.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "legal_uncertainty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Recent case law has established favorable precedents for claims like this.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "established_precedent"
        ),
        Statement(
            "Documented communications provide clear evidence of the agreement terms.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "documented_evidence"
        ),
    ]
)


# =============================================================================
# STIMULUS 12: Environmental Impact Assessment
# =============================================================================

STIMULUS_12 = SelectionStimulus(
    id="environmental_impact_12",
    context="You are reviewing an environmental assessment of a region to advise on conservation priorities.",
    instruction="Select the statement most important for determining conservation priorities.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "Restoration efforts have exceeded biodiversity targets in pilot areas.",
            AppraisalDimension.VALENCE_POSITIVE,
            "restoration_success"
        ),
        Statement(
            "Water quality indicators have improved consistently over the past five years.",
            AppraisalDimension.VALENCE_POSITIVE,
            "quality_improvement"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Three endemic species have declined below viable population thresholds.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "species_decline"
        ),
        Statement(
            "Wetland habitat destroyed in the 1990s cannot be restored to original function.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "irreversible_loss"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Targeted interventions have successfully stabilized at-risk populations before.",
            AppraisalDimension.AGENCY_HIGH,
            "intervention_track_record"
        ),
        Statement(
            "Landowner partnerships enable direct management of critical habitat corridors.",
            AppraisalDimension.AGENCY_HIGH,
            "management_access"
        ),
        
        # AGENCY_LOW
        Statement(
            "Invasive species introduction could trigger cascading effects that cannot be modeled.",
            AppraisalDimension.AGENCY_LOW,
            "invasive_threat"
        ),
        Statement(
            "Climate impacts are driven by global factors beyond regional management control.",
            AppraisalDimension.AGENCY_LOW,
            "climate_dependency"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Climate models show high variance in projected impacts for this ecosystem.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "climate_uncertainty"
        ),
        Statement(
            "The interaction between multiple stressors creates unpredictable compound risks.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "interaction_uncertainty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Long-term monitoring data provides reliable baselines for measuring change.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "monitoring_baseline"
        ),
        Statement(
            "Population dynamics for key species are well-understood from decades of research.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "species_knowledge"
        ),
    ]
)


# =============================================================================
# STIMULUS 13: Relationship Continuation Decision
# =============================================================================

STIMULUS_13 = SelectionStimulus(
    id="relationship_decision_13",
    context="You are helping a friend think through whether to continue a long-term romantic relationship.",
    instruction="Select the statement most important for making this relationship decision.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "Both partners have demonstrated capacity for growth during difficult periods.",
            AppraisalDimension.VALENCE_POSITIVE,
            "growth_capacity"
        ),
        Statement(
            "Recent efforts at communication have produced meaningful improvements.",
            AppraisalDimension.VALENCE_POSITIVE,
            "positive_trajectory"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Trust was violated when the partner concealed significant financial decisions.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "trust_violation"
        ),
        Statement(
            "The early connection and shared dreams have faded and cannot be recaptured.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "connection_loss"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Couples therapy has provided tools that both partners are actively using.",
            AppraisalDimension.AGENCY_HIGH,
            "therapeutic_tools"
        ),
        Statement(
            "Clear conversations have established specific changes each person will make.",
            AppraisalDimension.AGENCY_HIGH,
            "actionable_commitments"
        ),
        
        # AGENCY_LOW
        Statement(
            "Life circumstances could change in ways that would alter the relationship dynamics.",
            AppraisalDimension.AGENCY_LOW,
            "circumstantial_factors"
        ),
        Statement(
            "Extended family pressures affect the relationship in ways neither can fully control.",
            AppraisalDimension.AGENCY_LOW,
            "family_dynamics"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Neither person can know how they will feel about this decision years from now.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "future_feelings_unknown"
        ),
        Statement(
            "It is impossible to predict whether current problems will improve or worsen.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "trajectory_uncertainty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Shared values and life goals remain strongly aligned despite current tensions.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "value_alignment"
        ),
        Statement(
            "Behavioral patterns over years provide clear evidence of each person's tendencies.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "established_patterns"
        ),
    ]
)


# =============================================================================
# STIMULUS 14: Graduate Program Selection
# =============================================================================

STIMULUS_14 = SelectionStimulus(
    id="graduate_program_14",
    context="You are advising a student choosing between graduate programs in their field.",
    instruction="Select the statement most important for making this program decision.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "Graduates have established successful careers across diverse sectors.",
            AppraisalDimension.VALENCE_POSITIVE,
            "career_outcomes"
        ),
        Statement(
            "New interdisciplinary initiatives align well with emerging career paths.",
            AppraisalDimension.VALENCE_POSITIVE,
            "program_alignment"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "The program's reputation has declined from its peak ranking a decade ago.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "reputation_decline"
        ),
        Statement(
            "Several distinguished faculty who made the program attractive have since retired.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "faculty_loss"
        ),
        
        # AGENCY_HIGH
        Statement(
            "The program offers flexibility to customize coursework to individual interests.",
            AppraisalDimension.AGENCY_HIGH,
            "curricular_flexibility"
        ),
        Statement(
            "Students can switch advisors if initial mentorship relationships don't work.",
            AppraisalDimension.AGENCY_HIGH,
            "advisor_choice"
        ),
        
        # AGENCY_LOW
        Statement(
            "Funding availability beyond the first year is not guaranteed.",
            AppraisalDimension.AGENCY_LOW,
            "funding_uncertainty"
        ),
        Statement(
            "The field is evolving rapidly in ways that could change program relevance.",
            AppraisalDimension.AGENCY_LOW,
            "field_evolution"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Employment outcomes for graduates vary widely and are difficult to predict.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "outcome_variability"
        ),
        Statement(
            "Current students report that advertised opportunities do not always materialize.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "opportunity_uncertainty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Published placement data shows consistent employment rates over ten years.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "placement_data"
        ),
        Statement(
            "Student satisfaction and completion rates have been stable in recent cohorts.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "satisfaction_consistency"
        ),
    ]
)


# =============================================================================
# STIMULUS 15: Travel Safety Assessment
# =============================================================================

STIMULUS_15 = SelectionStimulus(
    id="travel_safety_15",
    context="You are advising someone planning an extended trip to a region they haven't visited before.",
    instruction="Select the statement most important for planning this trip.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "Travelers consistently report transformative experiences unavailable elsewhere.",
            AppraisalDimension.VALENCE_POSITIVE,
            "unique_experience"
        ),
        Statement(
            "Currency exchange rates currently offer exceptional value for this destination.",
            AppraisalDimension.VALENCE_POSITIVE,
            "value_proposition"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Many historical sites have deteriorated significantly from their documented condition.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "site_deterioration"
        ),
        Statement(
            "Traditional cultural practices that once attracted visitors have largely disappeared.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "cultural_loss"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Detailed itinerary planning can avoid most commonly reported problems.",
            AppraisalDimension.AGENCY_HIGH,
            "planning_mitigation"
        ),
        Statement(
            "Travel insurance options cover the specific risks associated with this region.",
            AppraisalDimension.AGENCY_HIGH,
            "insurance_coverage"
        ),
        
        # AGENCY_LOW
        Statement(
            "Political conditions in the region have been volatile with sudden changes.",
            AppraisalDimension.AGENCY_LOW,
            "political_volatility"
        ),
        Statement(
            "Healthcare quality varies significantly and emergency services are inconsistent.",
            AppraisalDimension.AGENCY_LOW,
            "healthcare_variability"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Local conditions can change rapidly with limited advance warning to travelers.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "conditions_unpredictable"
        ),
        Statement(
            "Information about the region online is often outdated or contradictory.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "information_unreliable"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Recent traveler reports provide current, detailed accounts of conditions.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "current_information"
        ),
        Statement(
            "Local communities have developed excellent, well-documented infrastructure for visitors.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "established_infrastructure"
        ),
    ]
)


# =============================================================================
# STIMULUS 16: Team Performance Analysis
# =============================================================================

STIMULUS_16 = SelectionStimulus(
    id="team_performance_16",
    context="You are analyzing why a previously successful team has underperformed this season.",
    instruction="Select the statement most important for understanding the team's performance.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "Young players have shown development exceeding initial projections.",
            AppraisalDimension.VALENCE_POSITIVE,
            "player_development"
        ),
        Statement(
            "Recent performances suggest the team is adapting to new strategies effectively.",
            AppraisalDimension.VALENCE_POSITIVE,
            "strategic_adaptation"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "The core players who defined the team's identity are past their peak years.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "peak_decline"
        ),
        Statement(
            "Departures of key contributors have left gaps that cannot be adequately filled.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "talent_loss"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Coaching staff has implemented specific adjustments that have shown early results.",
            AppraisalDimension.AGENCY_HIGH,
            "coaching_adjustments"
        ),
        Statement(
            "Draft position and cap space create opportunities for rapid rebuilding.",
            AppraisalDimension.AGENCY_HIGH,
            "rebuild_resources"
        ),
        
        # AGENCY_LOW
        Statement(
            "External factors affecting the team are beyond anyone's ability to control.",
            AppraisalDimension.AGENCY_LOW,
            "external_factors"
        ),
        Statement(
            "Injuries have affected key players at unpredictable times throughout the season.",
            AppraisalDimension.AGENCY_LOW,
            "injury_impact"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Performance fluctuations have no clear pattern that would enable prediction.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "pattern_uncertainty"
        ),
        Statement(
            "The underlying causes of the slump remain unclear despite extensive analysis.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "causal_uncertainty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Statistical analysis clearly identifies specific areas of performance decline.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "diagnostic_clarity"
        ),
        Statement(
            "Comparison with successful teams reveals specific, measurable gaps.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "benchmark_clarity"
        ),
    ]
)


# =============================================================================
# STIMULUS 17: Housing Purchase Decision
# =============================================================================

STIMULUS_17 = SelectionStimulus(
    id="housing_purchase_17",
    context="You are advising someone considering purchasing a home in a new neighborhood.",
    instruction="Select the statement most important for making this purchase decision.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "The property offers more space and features than comparable listings.",
            AppraisalDimension.VALENCE_POSITIVE,
            "comparative_value"
        ),
        Statement(
            "Recent sales in the neighborhood show steady appreciation.",
            AppraisalDimension.VALENCE_POSITIVE,
            "appreciation_trend"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "The seller failed to disclose known issues that were later discovered.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "disclosure_failure"
        ),
        Statement(
            "The neighborhood's most desirable features have diminished over time.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "neighborhood_decline"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Identified repairs can be negotiated into the purchase price.",
            AppraisalDimension.AGENCY_HIGH,
            "negotiation_leverage"
        ),
        Statement(
            "The buyer can specify inspection contingencies protecting against major issues.",
            AppraisalDimension.AGENCY_HIGH,
            "contingency_protection"
        ),
        
        # AGENCY_LOW
        Statement(
            "Planned developments nearby could affect the property in unpredictable ways.",
            AppraisalDimension.AGENCY_LOW,
            "development_exposure"
        ),
        Statement(
            "Property values in the area depend heavily on the regional economy.",
            AppraisalDimension.AGENCY_LOW,
            "economic_dependency"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Some structural elements could not be fully inspected and remain unknown.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "inspection_limitations"
        ),
        Statement(
            "Property values in the area have fluctuated significantly with no clear trend.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "value_uncertainty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Comprehensive inspection reports detail the exact condition of all systems.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "inspection_completeness"
        ),
        Statement(
            "Title search confirms clear ownership with no liens or encumbrances.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "title_clarity"
        ),
    ]
)


# =============================================================================
# STIMULUS 18: Career Transition Assessment
# =============================================================================

STIMULUS_18 = SelectionStimulus(
    id="career_transition_18",
    context="You are advising someone considering leaving their current career to pursue a different field.",
    instruction="Select the statement most important for making this career decision.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "The new field aligns better with long-held values and interests.",
            AppraisalDimension.VALENCE_POSITIVE,
            "value_alignment"
        ),
        Statement(
            "Others who made similar transitions report high satisfaction with the change.",
            AppraisalDimension.VALENCE_POSITIVE,
            "transition_outcomes"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Years of building expertise and reputation in the current field would be left behind.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "expertise_loss"
        ),
        Statement(
            "Professional relationships cultivated over a career cannot be replicated.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "network_loss"
        ),
        
        # AGENCY_HIGH
        Statement(
            "A structured retraining program provides a clear path to qualification.",
            AppraisalDimension.AGENCY_HIGH,
            "pathway_clarity"
        ),
        Statement(
            "Savings provide runway to pursue the transition without immediate income pressure.",
            AppraisalDimension.AGENCY_HIGH,
            "financial_buffer"
        ),
        
        # AGENCY_LOW
        Statement(
            "The new industry is experiencing disruption with unclear long-term stability.",
            AppraisalDimension.AGENCY_LOW,
            "industry_disruption"
        ),
        Statement(
            "Entry into the new field depends on hiring decisions by others.",
            AppraisalDimension.AGENCY_LOW,
            "hiring_dependency"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Success in the new field depends on factors that cannot be assessed in advance.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "success_uncertainty"
        ),
        Statement(
            "Skills that transfer well in theory may not translate in practice.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "transfer_uncertainty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Informational interviews have clarified exactly what the transition requires.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "requirement_clarity"
        ),
        Statement(
            "Transferable skills are in documented high demand in the target industry.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "demand_evidence"
        ),
    ]
)


# =============================================================================
# STIMULUS 19: Community Safety Assessment
# =============================================================================

STIMULUS_19 = SelectionStimulus(
    id="community_safety_19",
    context="You are helping someone evaluate a neighborhood they are considering moving to.",
    instruction="Select the statement most important for evaluating this neighborhood.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "New businesses opening indicate growing confidence in the area's future.",
            AppraisalDimension.VALENCE_POSITIVE,
            "business_investment"
        ),
        Statement(
            "Active community organizations have successfully advocated for improvements.",
            AppraisalDimension.VALENCE_POSITIVE,
            "community_engagement"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Long-term residents describe significant decline from the neighborhood's peak.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "resident_perspective"
        ),
        Statement(
            "Community institutions that once anchored the area have closed permanently.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "institutional_loss"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Homeowner associations provide mechanisms for addressing neighborhood issues.",
            AppraisalDimension.AGENCY_HIGH,
            "governance_structure"
        ),
        Statement(
            "Neighborhood watch programs give residents tools to address safety concerns.",
            AppraisalDimension.AGENCY_HIGH,
            "safety_tools"
        ),
        
        # AGENCY_LOW
        Statement(
            "Residents report that complaints to authorities are routinely ignored.",
            AppraisalDimension.AGENCY_LOW,
            "authority_unresponsive"
        ),
        Statement(
            "Economic forces affecting the area are driven by regional and national trends.",
            AppraisalDimension.AGENCY_LOW,
            "macro_dependency"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Crime patterns in the area are erratic with no clear safe zones or times.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "crime_unpredictability"
        ),
        Statement(
            "Environmental hazards have been identified but their health impacts are unclear.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "environmental_uncertainty"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Published crime statistics show consistent patterns over the past five years.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "crime_data"
        ),
        Statement(
            "Recent infrastructure investments have been documented with completion timelines.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "investment_documentation"
        ),
    ]
)


# =============================================================================
# STIMULUS 20: Organizational Restructuring
# =============================================================================

STIMULUS_20 = SelectionStimulus(
    id="org_restructuring_20",
    context="You are evaluating a proposed organizational restructuring at a company.",
    instruction="Select the statement most important for evaluating this restructuring plan.",
    statements=[
        # VALENCE_POSITIVE
        Statement(
            "The new structure eliminates redundancies that have frustrated employees.",
            AppraisalDimension.VALENCE_POSITIVE,
            "efficiency_gain"
        ),
        Statement(
            "Early transitions to the new model have shown productivity improvements.",
            AppraisalDimension.VALENCE_POSITIVE,
            "pilot_success"
        ),
        
        # VALENCE_NEGATIVE
        Statement(
            "Effective teams that took years to build will be disbanded.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "team_dissolution"
        ),
        Statement(
            "Institutional knowledge held by departing employees cannot be transferred.",
            AppraisalDimension.VALENCE_NEGATIVE,
            "knowledge_loss"
        ),
        
        # AGENCY_HIGH
        Statement(
            "Employees can apply for positions in the new structure matching their skills.",
            AppraisalDimension.AGENCY_HIGH,
            "employee_agency"
        ),
        Statement(
            "Feedback mechanisms allow affected staff to influence implementation details.",
            AppraisalDimension.AGENCY_HIGH,
            "input_channels"
        ),
        
        # AGENCY_LOW
        Statement(
            "The timeline depends on factors that management cannot fully control.",
            AppraisalDimension.AGENCY_LOW,
            "timeline_dependency"
        ),
        Statement(
            "Market conditions may force additional changes beyond what is currently planned.",
            AppraisalDimension.AGENCY_LOW,
            "market_pressure"
        ),
        
        # UNCERTAINTY_HIGH
        Statement(
            "Similar restructurings at other companies have had highly variable outcomes.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "outcome_variability"
        ),
        Statement(
            "Hidden interdependencies may cause unexpected problems during transition.",
            AppraisalDimension.UNCERTAINTY_HIGH,
            "interdependency_risk"
        ),
        
        # UNCERTAINTY_LOW
        Statement(
            "Change management best practices from successful transitions have been incorporated.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "best_practices"
        ),
        Statement(
            "Detailed transition plans specify exactly what changes occur and when.",
            AppraisalDimension.UNCERTAINTY_LOW,
            "plan_specificity"
        ),
    ]
)


# =============================================================================
# COLLECTION OF ALL STIMULI
# =============================================================================

ALL_STIMULI = [
    STIMULUS_1,
    STIMULUS_2,
    STIMULUS_3,
    STIMULUS_4,
    STIMULUS_5,
    STIMULUS_6,
    STIMULUS_7,
    STIMULUS_8,
    STIMULUS_9,
    STIMULUS_10,
    STIMULUS_11,
    STIMULUS_12,
    STIMULUS_13,
    STIMULUS_14,
    STIMULUS_15,
    STIMULUS_16,
    STIMULUS_17,
    STIMULUS_18,
    STIMULUS_19,
    STIMULUS_20,
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def run_single_trial(
    stimulus: SelectionStimulus,
    model_response_fn,
    seed: int = None
) -> Dict:
    """
    Run a single trial and return scored results.
    """
    prompt, mapping = stimulus.get_shuffled_prompt(seed=seed)
    selection_str = model_response_fn(prompt)
    result = stimulus.score_selection(selection_str, mapping)
    
    return {
        "stimulus_id": stimulus.id,
        "seed": seed,
        "prompt": prompt,
        "mapping": {k: {"text": v.text, "dimension": v.dimension.value} 
                    for k, v in mapping.items()},
        "selection": selection_str,
        "result": {
            "is_valid": result["is_valid"],
            "dimension": result["selected_dimension"].value if result["selected_dimension"] else None,
            "subcategory": result["selected_subcategory"]
        }
    }


def parse_selection_response(response: str) -> str:
    """
    Parse a model's response to extract a single letter (A, B, C...).
    """
    match = re.search(r'[A-L]', response.upper())
    if match:
        return match.group(0)
    return ""


def get_dimension_summary(results: List[Dict]) -> Dict[str, int]:
    """
    Summarize selections by appraisal dimension.
    """
    summary = {dim.value: 0 for dim in AppraisalDimension}
    for r in results:
        if r["result"]["is_valid"] and r["result"]["dimension"]:
            summary[r["result"]["dimension"]] += 1
    return summary


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=== Appraisal Theory Stimulus Set ===\n")
    
    print("Appraisal Dimensions:")
    for dim in AppraisalDimension:
        print(f"  - {dim.value}")
    
    print("\n=== Example Prompt ===")
    prompt, mapping = STIMULUS_1.get_shuffled_prompt(seed=42)
    print(prompt)
    
    print("\n=== Internal Mapping ===")
    for label, stmt in mapping.items():
        print(f"[{label}] {stmt.dimension.value}: {stmt.text[:60]}...")
    
    print("\n=== Example Scoring ===")
    example_response = "B"
    parsed_letter = parse_selection_response(example_response)
    score_result = STIMULUS_1.score_selection(parsed_letter, mapping)
    
    print(f"Response: '{example_response}'")
    print(f"Parsed: {parsed_letter}")
    print(f"Dimension: {score_result['selected_dimension'].value if score_result['selected_dimension'] else 'Invalid'}")
