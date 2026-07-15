export interface StepDetail {
  step_number: number;
  description: string;
  expression: string | null;
  expression_latex: string | null;
  result: string | null;
  result_latex: string | null;
  justification: string | null;
}

export interface AlternativeMethod {
  name: string;
  steps: string[];
  final_answer: string | null;
  final_answer_latex: string | null;
}

export interface FormulaUsed {
  name: string;
  formula_latex: string;
  description: string | null;
}

export interface GraphData {
  graph_type: string;
  data: Record<string, unknown>;
  plotly_json: Record<string, unknown> | null;
}

export interface MathContext {
  normalized_expression: string;
  latex: string | null;
  tokens: string[];
  operation: string;
  ast: string | null;
  variables: string[];
  selected_method: string | null;
  previous_variables: string[];
  previous_answer: string | null;
}

export interface SolveRequest {
  expression: string;
  topic_hint: string | null;
  include_graph: boolean;
  session_id?: string | null;
}

export interface SolveResponse {
  question: string;
  question_latex: string | null;
  extracted_text?: string | null;
  ocr_confidence?: number | null;
  topic: string;
  answer: string;
  answer_latex: string | null;
  steps: StepDetail[];
  difficulty: string | null;
  alternative_methods: AlternativeMethod[];
  formulas_used: FormulaUsed[];
  verification: string | null;
  verification_latex: string | null;
  graph_data: GraphData | null;
  math_context: MathContext | null;
  ai_confidence: number;
}

export interface VerifyRequest {
  expression: string;
  student_solution: string;
}

export interface VerificationStep {
  step_number: number;
  is_correct: boolean;
  student_step: string;
  feedback: string;
  correction: string | null;
}

export interface VerifyResponse {
  original_question: string;
  student_solution: string;
  is_correct: boolean;
  final_answer_correct: boolean;
  steps_analysis: VerificationStep[];
  overall_feedback: string;
  score: number;
}

export interface Question {
  id: string;
  question: string;
  question_latex: string | null;
  answer: string;
  answer_latex: string | null;
  steps: StepDetail[];
  topic: string | null;
  difficulty: string | null;
  tags: string[];
  alternative_methods: AlternativeMethod[];
  formulas_used: FormulaUsed[];
  ai_confidence: number | null;
  graph_data: GraphData | null;
  is_favorite: boolean;
  is_pinned: boolean;
  is_archived: boolean;
  view_count: number;
  notes: string | null;
  collection_id: string | null;
  collection_name: string | null;
  created_at: string;
  updated_at: string;
}

export interface QuestionListResponse {
  items: Question[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface Collection {
  id: string;
  name: string;
  description: string | null;
  parent_id: string | null;
  color_label: string | null;
  icon: string | null;
  sort_order: number;
  question_count: number;
  created_at: string;
  updated_at: string;
}

export interface QuestionStats {
  total_questions: number;
  questions_per_topic: Record<string, number>;
  difficulty_distribution: Record<string, number>;
  recently_added: Question[];
  recently_viewed: Question[];
  most_solved_topics: [string, number][];
}
