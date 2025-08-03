"""
Enhanced intelligent CLI for medical triage with contextual questioning,
disease-specific follow-ups, and improved user interaction.
"""

import joblib
import numpy as np
import pandas as pd
import rich
from rich.prompt import Prompt, Confirm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
from sklearn.tree import _tree
from sklearn.feature_selection import mutual_info_classif
from preprocessor import get_symptom_severity_score, get_demographic_risk_factors

console = Console()

# Load enhanced model bundle
try:
    MODEL_BUNDLE = joblib.load(Path("models/triage_model_enhanced.joblib"))
    MODEL = MODEL_BUNDLE["model"]
    DEMO_PIPE = MODEL_BUNDLE["demo_pipe"]
    MLB = MODEL_BUNDLE["mlb"]
    FEATURE_SELECTOR = MODEL_BUNDLE["feature_selector"]
    SELECTED_FEATURES = MODEL_BUNDLE["selected_features"]
    console.print("✅ Enhanced model loaded", style="green")
except FileNotFoundError:
    # Fallback to legacy model
    MODEL_BUNDLE = joblib.load(Path("models/triage_model.joblib"))
    MODEL = MODEL_BUNDLE["tree"]
    DEMO_PIPE = MODEL_BUNDLE["demo_pipe"]
    MLB = MODEL_BUNDLE["mlb"]
    FEATURE_SELECTOR = None
    SELECTED_FEATURES = None
    console.print("⚠️ Using legacy model (run training to get enhanced features)", style="yellow")

SEVERITY = MODEL_BUNDLE["severity_map"]
DESC_DF = MODEL_BUNDLE["desc_df"].set_index("Disease")
PREC_DF = MODEL_BUNDLE["prec_df"].set_index("Disease")
FEATURE_NAMES = MODEL_BUNDLE["feature_names"]

SYMPTOMS = MLB.classes_.tolist()
DEMOGRAPHICS = ["age", "height_cm", "weight_kg", "gender"]

# Enhanced symptom descriptions for better user understanding
SYMPTOM_DESCRIPTIONS = {
    'fever': 'elevated body temperature (feeling hot, chills, sweating)',
    'headache': 'pain or pressure in your head or neck area',
    'cough': 'persistent coughing or throat clearing',
    'fatigue': 'unusual tiredness or weakness',
    'nausea': 'feeling sick to your stomach or queasy',
    'vomiting': 'throwing up or retching',
    'diarrhea': 'loose or watery bowel movements',
    'abdominal_pain': 'pain or discomfort in your stomach area',
    'chest_pain': 'pain or pressure in your chest',
    'breathlessness': 'difficulty breathing or shortness of breath',
    'dizziness': 'feeling lightheaded or unsteady',
    'joint_pain': 'aching or stiffness in your joints',
    'muscle_pain': 'soreness or aching in your muscles',
    'rash': 'skin irritation, bumps, or unusual marks on skin',
    'sore_throat': 'pain or scratchiness when swallowing'
}

# Disease context for intelligent follow-up questions
DISEASE_CONTEXT = {
    'respiratory': {
        'keywords': ['cough', 'breathlessness', 'chest_pain', 'fever'],
        'follow_ups': [
            "Are you experiencing any difficulty breathing even at rest?",
            "Do you have any chest tightness or wheezing?",
            "Have you been exposed to anyone with respiratory illness recently?",
            "Are you coughing up any phlegm or blood?"
        ]
    },
    'gastrointestinal': {
        'keywords': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain'],
        'follow_ups': [
            "Have you eaten anything unusual in the past 24 hours?",
            "Are you experiencing any blood in stool or vomit?",
            "Do you have severe dehydration (dry mouth, dizziness)?",
            "Have you had similar episodes before?"
        ]
    },
    'neurological': {
        'keywords': ['headache', 'dizziness', 'fatigue'],
        'follow_ups': [
            "Is this the worst headache you've ever experienced?",
            "Do you have any vision changes or sensitivity to light?",
            "Are you experiencing any confusion or memory problems?",
            "Have you had any recent head injuries?"
        ]
    },
    'infectious': {
        'keywords': ['fever', 'fatigue', 'muscle_pain', 'sore_throat'],
        'follow_ups': [
            "Have you traveled recently or been exposed to sick individuals?",
            "Do you have any swollen lymph nodes?",
            "Are you experiencing night sweats or chills?",
            "Have you had any recent vaccinations?"
        ]
    }
}

def ask_symptom_with_context(symptom: str, context: str = None) -> int:
    """Ask about a symptom with enhanced description and context."""
    base_desc = SYMPTOM_DESCRIPTIONS.get(symptom, symptom.replace("_", " "))
    
    if context:
        question = f"Given {context}, do you have **{base_desc}**?"
    else:
        question = f"Do you experience **{base_desc}**?"
    
    console.print(f"\n[cyan]❓ {question}[/cyan]")
    
    # Provide severity context for important symptoms
    if symptom in SEVERITY and SEVERITY[symptom] >= 4:
        console.print(f"[yellow]⚠️ This is considered a significant symptom[/yellow]")
    
    ans = Prompt.ask(
        "Response",
        choices=["yes", "no", "unsure", "help"], 
        default="unsure"
    )
    
    if ans == "help":
        console.print(f"[dim]Help: {base_desc}[/dim]")
        ans = Prompt.ask(
            "Response", 
            choices=["yes", "no", "unsure"], 
            default="unsure"
        )
    
    return {"yes": 1, "no": 0, "unsure": -1}[ans]

def get_contextual_questions(current_symptoms: list) -> list:
    """Generate contextual follow-up questions based on symptom patterns."""
    questions = []
    
    for category, data in DISEASE_CONTEXT.items():
        symptom_overlap = set(current_symptoms) & set(data['keywords'])
        if len(symptom_overlap) >= 2:  # If multiple symptoms from same category
            questions.extend(data['follow_ups'][:2])  # Add top 2 follow-ups
    
    return questions[:3]  # Limit to 3 follow-up questions

def calculate_risk_score(demographics: dict, symptoms: list) -> dict:
    """Calculate comprehensive risk assessment."""
    risk_score = 0
    risk_factors = []
    
    # Demographic risk factors
    age = demographics.get('Age', 30)
    height = demographics.get('Height_cm', 170)
    weight = demographics.get('Weight_kg', 70)
    gender = demographics.get('Gender', 'unknown')
    
    bmi = weight / ((height / 100) ** 2) if height > 0 else 25
    demo_risks = get_demographic_risk_factors(age, bmi, gender)
    risk_factors.extend(demo_risks)
    
    # Age-based risk
    if age >= 65:
        risk_score += 2
    elif age <= 5:
        risk_score += 3
    
    # Symptom severity risk
    severity_score = get_symptom_severity_score(symptoms, SEVERITY)
    risk_score += severity_score / 10  # Normalize
    
    # High-risk symptom combinations
    high_risk_combos = [
        ['chest_pain', 'breathlessness'],
        ['fever', 'headache', 'rash'],
        ['severe_headache', 'vision_changes'],
        ['abdominal_pain', 'vomiting', 'fever']
    ]
    
    for combo in high_risk_combos:
        if len(set(combo) & set(symptoms)) >= len(combo) - 1:
            risk_score += 3
            risk_factors.append(f"High-risk combination: {', '.join(combo)}")
    
    return {
        'score': risk_score,
        'level': 'HIGH' if risk_score >= 6 else 'MEDIUM' if risk_score >= 3 else 'LOW',
        'factors': risk_factors
    }

def ask_follow_up_questions(predicted_diseases: list, current_symptoms: list):
    """Ask intelligent follow-up questions based on predicted diseases."""
    console.print("\n[bold blue]🔍 Additional Questions[/bold blue]")
    
    # Get disease-specific questions
    contextual_questions = get_contextual_questions(current_symptoms)
    
    # Get precaution-based questions
    precaution_questions = []
    for disease, prob in predicted_diseases[:2]:  # Top 2 diseases
        if disease in PREC_DF.index:
            precautions = PREC_DF.loc[disease].dropna().tolist()
            if precautions:
                precaution_questions.append(
                    f"For {disease}: Have you tried {precautions[0].lower()}?"
                )
    
    all_questions = contextual_questions + precaution_questions
    
    responses = []
    for question in all_questions[:3]:  # Limit to 3 questions
        response = Confirm.ask(f"[cyan]{question}[/cyan]")
        responses.append((question, response))
    
    return responses

def intelligent_next_question(known_vec, unknown_syms, current_symptoms: list):
    """Enhanced symptom selection using multiple criteria."""
    if not unknown_syms:
        return None
    
    # Create dummy data for mutual information calculation
    X_dummy = np.vstack([known_vec, known_vec])
    y_dummy = np.array([0, 1])
    
    try:
        gains = mutual_info_classif(X_dummy, y_dummy, discrete_features=True)
        gain_map = {SYMPTOMS[i]: gains[len(DEMOGRAPHICS) + i] for i in range(len(SYMPTOMS))}
    except:
        # Fallback to uniform gains
        gain_map = {sym: 1.0 for sym in SYMPTOMS}
    
    scores = {}
    for sym in unknown_syms:
        score = 0
        
        # Information gain component
        score += gain_map.get(sym, 0.0) * 3
        
        # Severity component
        score += (SEVERITY.get(sym, 3) / 5) * 2
        
        # Context relevance (if symptom relates to current symptoms)
        for category, data in DISEASE_CONTEXT.items():
            if sym in data['keywords'] and any(s in current_symptoms for s in data['keywords']):
                score += 1.5
        
        # Common symptom interactions
        interaction_bonus = 0
        if 'fever' in current_symptoms and sym in ['headache', 'fatigue', 'muscle_pain']:
            interaction_bonus += 1
        if 'cough' in current_symptoms and sym in ['fever', 'chest_pain', 'breathlessness']:
            interaction_bonus += 1
        if 'nausea' in current_symptoms and sym in ['vomiting', 'abdominal_pain', 'diarrhea']:
            interaction_bonus += 1
        
        score += interaction_bonus
        scores[sym] = score
    
    return max(scores, key=scores.get)

def display_results(diseases: list, demographics: dict, symptoms: list, follow_up_responses: list):
    """Enhanced results display with comprehensive information."""
    
    # Calculate risk assessment
    risk_assessment = calculate_risk_score(demographics, symptoms)
    
    # Create results table
    table = Table(title="🏥 Medical Triage Results")
    table.add_column("Disease", style="cyan", width=20)
    table.add_column("Probability", style="green", justify="center")
    table.add_column("Description", style="dim", width=40)
    
    for disease, prob in diseases:
        desc = DESC_DF.at[disease, 'Description'] if disease in DESC_DF.index else "No description available"
        table.add_row(
            disease,
            f"{prob:.1%}",
            desc[:80] + "..." if len(desc) > 80 else desc
        )
    
    console.print(table)
    
    # Display risk assessment
    risk_color = "red" if risk_assessment['level'] == 'HIGH' else "yellow" if risk_assessment['level'] == 'MEDIUM' else "green"
    console.print(f"\n[bold {risk_color}]🚨 Risk Level: {risk_assessment['level']} (Score: {risk_assessment['score']:.1f})[/bold {risk_color}]")
    
    if risk_assessment['factors']:
        console.print("\n[bold]Risk Factors:[/bold]")
        for factor in risk_assessment['factors']:
            console.print(f"  • {factor}")
    
    # Display precautions for top disease
    top_disease = diseases[0][0]
    if top_disease in PREC_DF.index:
        precautions = PREC_DF.loc[top_disease].dropna().tolist()
        if precautions:
            console.print(f"\n[bold blue]💡 Recommended Precautions for {top_disease}:[/bold blue]")
            for i, precaution in enumerate(precautions[:4], 1):
                console.print(f"  {i}. {precaution}")
    
    # Emergency warnings for high-risk conditions
    if risk_assessment['level'] == 'HIGH' or diseases[0][1] > 0.7:
        console.print(Panel(
            "[bold red]⚠️ IMPORTANT: These results are for informational purposes only. "
            "Consult a healthcare professional immediately if you have serious concerns.[/bold red]",
            border_style="red"
        ))

def enhanced_cli():
    """Enhanced CLI with intelligent questioning and comprehensive assessment."""
    console.print(Panel.fit(
        "[bold cyan]🏥 Enhanced Medical Triage AI[/bold cyan]\n"
        "[dim]Intelligent symptom assessment with contextual questioning[/dim]",
        border_style="cyan"
    ))
    
    # Gather demographics with validation
    console.print("\n[bold blue]📋 Patient Information[/bold blue]")
    
    demographics = {}
    demographics["Age"] = int(Prompt.ask("Age (years)", default="30"))
    demographics["Height_cm"] = float(Prompt.ask("Height (cm)", default="170"))
    demographics["Weight_kg"] = float(Prompt.ask("Weight (kg)", default="70"))
    demographics["Gender"] = Prompt.ask(
        "Gender", 
        choices=["male", "female", "other"], 
        default="male"
    )
    
    # Calculate and display BMI
    bmi = demographics["Weight_kg"] / ((demographics["Height_cm"] / 100) ** 2)
    console.print(f"[dim]Calculated BMI: {bmi:.1f}[/dim]")
    
    # Create enhanced demographic features that match training pipeline
    enhanced_demographics = demographics.copy()
    
    # Add derived features (same as in preprocessing)
    enhanced_demographics['BMI'] = bmi
    
    # Age groups
    age = demographics["Age"]
    if age <= 18:
        age_group = 'Child'
    elif age <= 30:
        age_group = 'Young Adult'
    elif age <= 50:
        age_group = 'Adult'
    elif age <= 70:
        age_group = 'Middle Age'
    else:
        age_group = 'Senior'
    enhanced_demographics['Age_Group'] = age_group
    
    # BMI categories
    if bmi < 18.5:
        bmi_category = 'Underweight'
    elif bmi < 25:
        bmi_category = 'Normal'
    elif bmi < 30:
        bmi_category = 'Overweight'
    else:
        bmi_category = 'Obese'
    enhanced_demographics['BMI_Category'] = bmi_category
    
    # Transform demographics with all required features
    demo_vec = DEMO_PIPE.transform(pd.DataFrame([enhanced_demographics]))
    symptom_vec = np.zeros((1, len(SYMPTOMS)))
    unknown_syms = set(SYMPTOMS)
    current_symptoms = []
    
    # Get initial complaint
    console.print("\n[bold blue]🩺 Symptom Assessment[/bold blue]")
    main_symptom = Prompt.ask(
        "What is your main concern or symptom?",
        default=""
    ).strip().lower().replace(" ", "_")
    
    if main_symptom in unknown_syms:
        symptom_vec[0, SYMPTOMS.index(main_symptom)] = 1
        unknown_syms.remove(main_symptom)
        current_symptoms.append(main_symptom)
        console.print(f"[green]✓ Noted: {main_symptom.replace('_', ' ')}[/green]")
    
    # Iterative intelligent questioning
    question_count = 0
    max_questions = 8  # Limit to prevent fatigue
    
    while question_count < max_questions:
        # Build feature vector with proper feature engineering
        demo_vec_array = demo_vec.toarray() if hasattr(demo_vec, 'toarray') else demo_vec
        
        # Create symptom severity features (matching training pipeline)
        if len(current_symptoms) > 0:
            severity_weights = np.array([SEVERITY.get(sym, 3.0) for sym in SYMPTOMS])
            severity_weights = severity_weights / severity_weights.max()
            symptom_vec_weighted = symptom_vec * severity_weights
        else:
            symptom_vec_weighted = symptom_vec
            
        # Add severity-based features (symptom_count, total_severity, avg_severity)
        symptom_count = symptom_vec.sum()
        total_severity = (symptom_vec * np.array([SEVERITY.get(sym, 3.0) for sym in SYMPTOMS])).sum()
        avg_severity = total_severity / symptom_count if symptom_count > 0 else 0
        severity_features = np.array([[symptom_count, total_severity, avg_severity]])
        
        # Build full feature vector (matching training pipeline structure)
        full_vec = np.hstack([demo_vec_array, symptom_vec_weighted, severity_features])
        
        # Apply feature selection if available
        if FEATURE_SELECTOR:
            try:
                full_vec_selected = FEATURE_SELECTOR.transform(full_vec)
            except Exception:
                # Fallback: use first 50 features if selector fails
                full_vec_selected = full_vec[:, :50] if full_vec.shape[1] >= 50 else full_vec
        else:
            # Use legacy model approach
            full_vec_selected = np.hstack([demo_vec_array, symptom_vec])
        
        # Get predictions
        try:
            probs = MODEL.predict_proba(full_vec_selected)[0]
            top_idx = probs.argsort()[::-1][:3]
            top_diseases = [(MODEL.classes_[i], probs[i]) for i in top_idx]
        except Exception as e:
            console.print(f"[red]Error in prediction: {e}[/red]")
            break
        
        # Stop conditions
        if probs[top_idx[0]] > 0.75 or not unknown_syms or question_count >= max_questions:
            break
        
        # Get next best question
        next_symptom = intelligent_next_question(
            full_vec_selected[0] if len(full_vec_selected.shape) > 1 else full_vec_selected,
            unknown_syms,
            current_symptoms
        )
        
        if not next_symptom:
            break
        
        # Ask the question with context
        context = f"considering your {', '.join(current_symptoms[:2])}" if current_symptoms else None
        answer = ask_symptom_with_context(next_symptom, context)
        
        if answer == -1:  # unsure
            unknown_syms.remove(next_symptom)
        else:
            symptom_vec[0, SYMPTOMS.index(next_symptom)] = answer
            unknown_syms.remove(next_symptom)
            if answer == 1:
                current_symptoms.append(next_symptom)
        
        question_count += 1
    
    # Get final predictions with proper feature engineering
    demo_vec_array = demo_vec.toarray() if hasattr(demo_vec, 'toarray') else demo_vec
    
    # Create final symptom severity features 
    if len(current_symptoms) > 0:
        severity_weights = np.array([SEVERITY.get(sym, 3.0) for sym in SYMPTOMS])
        severity_weights = severity_weights / severity_weights.max()
        symptom_vec_weighted = symptom_vec * severity_weights
    else:
        symptom_vec_weighted = symptom_vec
        
    # Add final severity-based features
    symptom_count = symptom_vec.sum()
    total_severity = (symptom_vec * np.array([SEVERITY.get(sym, 3.0) for sym in SYMPTOMS])).sum()
    avg_severity = total_severity / symptom_count if symptom_count > 0 else 0
    severity_features = np.array([[symptom_count, total_severity, avg_severity]])
    
    # Build final feature vector
    full_vec = np.hstack([demo_vec_array, symptom_vec_weighted, severity_features])
    
    # Apply feature selection
    if FEATURE_SELECTOR:
        try:
            full_vec_selected = FEATURE_SELECTOR.transform(full_vec)
        except Exception:
            full_vec_selected = full_vec[:, :50] if full_vec.shape[1] >= 50 else full_vec
    else:
        full_vec_selected = np.hstack([demo_vec_array, symptom_vec])
    
    try:
        probs = MODEL.predict_proba(full_vec_selected)[0]
        top_idx = probs.argsort()[::-1][:5]
        final_diseases = [(MODEL.classes_[i], probs[i]) for i in top_idx]
    except Exception as e:
        console.print(f"[red]Error in final prediction: {e}[/red]")
        return
    
    # Ask contextual follow-up questions
    follow_up_responses = ask_follow_up_questions(final_diseases, current_symptoms)
    
    # Display comprehensive results
    console.print("\n" + "="*60)
    display_results(final_diseases, enhanced_demographics, current_symptoms, follow_up_responses)
    
    # Confidence and recommendations
    top_confidence = final_diseases[0][1]
    if top_confidence < 0.3:
        console.print("\n[yellow]⚠️ Low confidence in diagnosis. Consider consulting a healthcare professional.[/yellow]")
    elif top_confidence > 0.8:
        console.print("\n[green]✅ High confidence in assessment.[/green]")
    
    console.print("\n[bold]🏥 Remember: This AI provides preliminary assessment only. Always consult healthcare professionals for medical decisions.[/bold]")

if __name__ == "__main__":
    enhanced_cli()