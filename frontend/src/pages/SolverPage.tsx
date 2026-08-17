import { useState, useRef } from 'react';
import toast from 'react-hot-toast';
import { api } from '../services/api';
import MathRenderer from '../components/MathRenderer';
import { SolveResponse, VerifyResponse } from '../utils/types';
import { Calculator, CheckCircle, ArrowRight, Lightbulb, BookOpen, Loader2, Upload, Image, X } from 'lucide-react';

const examples = [
  'integrate x^2 * sin(x) dx',
  'solve x^2 - 4 = 0',
  'simplify sin(x)^2 + cos(x)^2',
  'derivative of x^3 * e^x',
  'limit of sin(x)/x as x -> 0',
  'integrate 1/(x^2 + 1) dx',
  "y'' + 4*y = sin(x)",
  "ode y'' + y = 0 with y(0) = 1, y'(0) = 0",
  'taylor exp(x) order 4',
  'taylor log(x) at x=1 order 3',
  'convergence of sum 1/n from n=1 to oo',
  'convergence of sum n/2^n from n=1 to oo',
  'laplace exp(3*x)',
  'inverse laplace 1/(s^2 + 1)',
  'fourier x on [-pi, pi]',
  'gradient of x^2*y + z',
  'divergence of x*y, y*z, z*x',
  'curl of y*z, x*z, x*y',
  'laplacian of x^2 + y^2 + 3*z^2',
];

export default function SolverPage() {
  const [expression, setExpression] = useState('');
  const [mode, setMode] = useState<'solve' | 'verify'>('solve');
  const [studentSolution, setStudentSolution] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [verifyResult, setVerifyResult] = useState<VerifyResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'steps' | 'understanding' | 'alternatives' | 'formulas'>('steps');
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleSolve() {
    if (!expression.trim()) return;
    setLoading(true);
    setVerifyResult(null);
    try {
      const res = await api.solve({ expression: expression.trim(), include_graph: true, session_id: 'default' });
      setResult(res);
      toast.success('Solved!');
    } catch (e: any) {
      toast.error(e.message || 'Solve failed');
    } finally {
      setLoading(false);
    }
  }

  async function handleVerify() {
    if (!expression.trim() || !studentSolution.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await api.verify({ expression: expression.trim(), student_solution: studentSolution.trim() });
      setVerifyResult(res);
      toast.success(res.is_correct ? 'Solution is correct!' : 'Solution needs review');
    } catch (e: any) {
      toast.error(e.message || 'Verification failed');
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload() {
    if (!uploadedImage) return;
    setUploadLoading(true);
    setVerifyResult(null);
    setResult(null);
    try {
      const res = await api.upload(uploadedImage);
      setResult(res);
      if (res.extracted_text) {
        setExpression(res.extracted_text);
      }
      clearImage();
      toast.success(res.topic !== 'Unknown' ? 'Image processed!' : 'No text extracted, try a clearer image');
    } catch (e: any) {
      toast.error(e.message || 'Upload failed');
    } finally {
      setUploadLoading(false);
    }
  }

  function handleImageSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      toast.error('Please select an image file');
      return;
    }
    setUploadedImage(file);
    const reader = new FileReader();
    reader.onload = () => setImagePreview(reader.result as string);
    reader.readAsDataURL(file);
  }

  function clearImage() {
    setUploadedImage(null);
    setImagePreview(null);
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Mode selector */}
      <div className="flex gap-2">
        {(['solve', 'verify'] as const).map(m => (
          <button
            key={m}
            onClick={() => { setMode(m); setResult(null); setVerifyResult(null); }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              mode === m
                ? 'bg-primary-500 text-white'
                : 'bg-[var(--card)] border border-[var(--border)] text-[var(--muted)] hover:text-[var(--fg)]'
            }`}
          >
            {m === 'solve' ? 'Solve' : 'Verify Solution'}
          </button>
        ))}
      </div>

      {/* Input area */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 space-y-4">
        <label className="block text-sm font-medium text-[var(--muted)]">
          {mode === 'solve' ? 'Enter a calculus problem' : 'Enter the original problem'}
        </label>
        <textarea
          value={expression}
          onChange={e => setExpression(e.target.value)}
          placeholder="e.g., integrate x^2 * sin(x) dx, solve x^2 - 4 = 0, simplify sin(x)^2 + cos(x)^2..."
          rows={3}
          className="w-full px-4 py-3 bg-[var(--bg)] border border-[var(--border)] rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none text-sm"
          onKeyDown={e => e.key === 'Enter' && e.ctrlKey && (mode === 'solve' ? handleSolve() : undefined)}
        />

        {/* Or upload an image */}
        <div className="border-t border-[var(--border)] pt-4">
          <label className="block text-sm font-medium text-[var(--muted)] mb-2">
            Or upload an image of the problem
          </label>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageSelect}
            className="hidden"
          />
          {!imagePreview ? (
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 px-4 py-3 border-2 border-dashed border-[var(--border)] rounded-lg text-sm text-[var(--muted)] hover:border-primary-500 hover:text-primary-500 transition-colors w-full justify-center"
            >
              <Image className="w-5 h-5" />
              Click to upload image (handwritten or printed)
            </button>
          ) : (
            <div className="flex items-start gap-3">
              <img
                src={imagePreview}
                alt="Preview"
                className="w-24 h-24 object-cover rounded-lg border border-[var(--border)]"
              />
              <div className="flex-1 space-y-2">
                <div className="text-xs text-[var(--muted)]">Image selected: {uploadedImage?.name}</div>
                <div className="flex gap-2">
                  <button
                    onClick={handleUpload}
                    disabled={uploadLoading}
                    className="flex items-center gap-2 px-4 py-2 bg-primary-500 text-white rounded-lg text-sm font-medium hover:bg-primary-600 disabled:opacity-50 transition-colors"
                  >
                    {uploadLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                    {uploadLoading ? 'Processing...' : 'Extract & Solve'}
                  </button>
                  <button
                    onClick={clearImage}
                    className="flex items-center gap-2 px-4 py-2 border border-[var(--border)] rounded-lg text-sm text-[var(--muted)] hover:text-[var(--fg)] transition-colors"
                  >
                    <X className="w-4 h-4" />
                    Remove
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {mode === 'verify' && (
          <>
            <label className="block text-sm font-medium text-[var(--muted)]">Student's solution</label>
            <textarea
              value={studentSolution}
              onChange={e => setStudentSolution(e.target.value)}
              placeholder="Paste the student's solution here..."
              rows={4}
              className="w-full px-4 py-3 bg-[var(--bg)] border border-[var(--border)] rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none text-sm"
            />
          </>
        )}

        <div className="flex gap-2 flex-wrap">
          {mode === 'solve' ? (
            <button
              onClick={handleSolve}
              disabled={loading || !expression.trim()}
              className="flex items-center gap-2 px-5 py-2.5 bg-primary-500 text-white rounded-lg font-medium text-sm hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Calculator className="w-4 h-4" />}
              Solve
            </button>
          ) : (
            <button
              onClick={handleVerify}
              disabled={loading || !expression.trim() || !studentSolution.trim()}
              className="flex items-center gap-2 px-5 py-2.5 bg-primary-500 text-white rounded-lg font-medium text-sm hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle className="w-4 h-4" />}
              Verify
            </button>
          )}
        </div>

        {/* Examples */}
        <div className="flex flex-wrap gap-2">
          <span className="text-xs text-[var(--muted)]">Try:</span>
          {examples.map(ex => (
            <button
              key={ex}
              onClick={() => setExpression(ex)}
              className="text-xs px-2 py-1 rounded-md bg-[var(--bg)] border border-[var(--border)] hover:border-primary-500 transition-colors text-[var(--muted)] hover:text-[var(--fg)]"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>

      {/* Solve result */}
      {result && (
        <div className="space-y-4 animate-in fade-in">
          {result.extracted_text !== undefined && (
            <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
              <div className="flex items-start justify-between gap-3 flex-wrap">
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium text-[var(--muted)] mb-2">Text extracted from image</div>
                  <div className="px-3 py-2 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-sm break-words">
                    {result.extracted_text || 'No text could be extracted.'}
                  </div>
                </div>
                {typeof result.ocr_confidence === 'number' && (
                  <div className="text-sm text-[var(--muted)]">
                    OCR confidence: {(result.ocr_confidence * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Header card */}
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
            <div className="flex items-start justify-between flex-wrap gap-2">
              <div>
                <div className="flex items-center gap-2 text-sm text-[var(--muted)]">
                  <BookOpen className="w-4 h-4" />
                  {result.topic}
                  <span className={`px-2 py-0.5 text-xs rounded-full ${
                    result.difficulty === 'Expert' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                    result.difficulty === 'Hard' ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400' :
                    result.difficulty === 'Medium' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
                    'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                  }`}>
                    {result.difficulty}
                  </span>
                </div>
                <div className="mt-1">
                  {result.question_latex ? (
                    <MathRenderer latex={result.question_latex} displayMode />
                  ) : (
                    <div className="text-sm mt-2 whitespace-pre-wrap break-words">{result.question}</div>
                  )}
                </div>
              </div>
              <div className="text-sm text-[var(--muted)]">
                Confidence: {(result.ai_confidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Answer */}
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl p-5">
            <div className="text-sm font-medium text-green-700 dark:text-green-400 mb-2">Answer</div>
            {result.answer_latex ? (
              <MathRenderer latex={result.answer_latex} displayMode />
            ) : (
              <div className="text-sm whitespace-pre-wrap break-words text-green-900 dark:text-green-100">
                {result.answer}
              </div>
            )}
          </div>

          {/* Verification */}
          {result.verification && (
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4 text-sm">
              <div className="flex items-center gap-2 font-medium text-blue-700 dark:text-blue-400 mb-1">
                <CheckCircle className="w-4 h-4" />
                Verification
              </div>
              <p className="text-blue-600 dark:text-blue-300">{result.verification}</p>
            </div>
          )}

          {/* Tabs: Steps / Alternatives / Formulas */}
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl overflow-hidden">
            <div className="flex border-b border-[var(--border)]">
              {[
                { key: 'steps', label: 'Step-by-Step', count: result.steps.length },
                { key: 'understanding', label: 'Understanding', count: result.math_context ? result.math_context.tokens.length : 0 },
                { key: 'alternatives', label: 'Alternative Methods', count: result.alternative_methods.length },
                { key: 'formulas', label: 'Formulas Used', count: result.formulas_used.length },
              ].map(tab => (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key as any)}
                  className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                    activeTab === tab.key
                      ? 'text-primary-600 dark:text-primary-400 border-b-2 border-primary-500'
                      : 'text-[var(--muted)] hover:text-[var(--fg)]'
                  }`}
                >
                  {tab.label} ({tab.count})
                </button>
              ))}
            </div>

            <div className="p-5 max-h-96 overflow-y-auto">
              {activeTab === 'steps' && (
                <div className="space-y-4">
                  {result.steps.map((step, i) => (
                    <div key={i} className="flex gap-3">
                      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-primary-500/10 flex items-center justify-center text-xs font-bold text-primary-600 dark:text-primary-400">
                        {step.step_number}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm">{step.description}</div>
                        {step.expression_latex && (
                          <div className="mt-1 p-2 bg-[var(--bg)] rounded-md">
                            <MathRenderer latex={step.expression_latex} displayMode />
                          </div>
                        )}
                        {step.justification && (
                          <div className="mt-1 flex items-start gap-1 text-xs text-[var(--muted)]">
                            <Lightbulb className="w-3 h-3 mt-0.5 flex-shrink-0" />
                            {step.justification}
                          </div>
                        )}
                        {step.result && i === result.steps.length - 1 && (
                          <div className="mt-2 p-2 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md">
                            <MathRenderer latex={step.result_latex || step.result} displayMode />
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'understanding' && (
                <div className="space-y-4">
                  {result.math_context ? (
                    <>
                      <div className="grid sm:grid-cols-2 gap-3">
                        <div className="p-3 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
                          <div className="text-xs font-medium text-[var(--muted)] mb-1">Operation</div>
                          <div className="text-sm">{result.math_context.operation}</div>
                        </div>
                        <div className="p-3 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
                          <div className="text-xs font-medium text-[var(--muted)] mb-1">Selected method</div>
                          <div className="text-sm">{result.math_context.selected_method || 'Symbolic computation'}</div>
                        </div>
                      </div>

                      <div className="p-3 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
                        <div className="text-xs font-medium text-[var(--muted)] mb-2">Normalized expression</div>
                        <div className="text-sm break-words">{result.math_context.normalized_expression}</div>
                        {result.math_context.latex && (
                          <div className="mt-2">
                            <MathRenderer latex={result.math_context.latex} displayMode />
                          </div>
                        )}
                      </div>

                      <div className="p-3 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
                        <div className="text-xs font-medium text-[var(--muted)] mb-2">Tokens</div>
                        <div className="flex flex-wrap gap-1.5">
                          {result.math_context.tokens.map((token, index) => (
                            <span key={`${token}-${index}`} className="px-2 py-1 rounded-md border border-[var(--border)] text-xs bg-[var(--card)]">
                              {token}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="grid sm:grid-cols-2 gap-3">
                        <div className="p-3 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
                          <div className="text-xs font-medium text-[var(--muted)] mb-1">Variables</div>
                          <div className="text-sm">{result.math_context.variables.join(', ') || 'None'}</div>
                        </div>
                        <div className="p-3 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
                          <div className="text-xs font-medium text-[var(--muted)] mb-1">Context memory</div>
                          <div className="text-sm break-words">
                            {result.math_context.previous_answer
                              ? `Previous: ${result.math_context.previous_answer}`
                              : 'No previous result in this session'}
                          </div>
                        </div>
                      </div>

                      {result.math_context.ast && (
                        <div className="p-3 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
                          <div className="text-xs font-medium text-[var(--muted)] mb-2">AST preview</div>
                          <pre className="text-xs whitespace-pre-wrap break-words font-mono text-[var(--muted)]">
                            {result.math_context.ast}
                          </pre>
                        </div>
                      )}
                    </>
                  ) : (
                    <p className="text-sm text-[var(--muted)] text-center py-8">No parsing context available.</p>
                  )}
                </div>
              )}

              {activeTab === 'alternatives' && (
                <div className="space-y-4">
                  {result.alternative_methods.map((method, i) => (
                    <div key={i} className="p-4 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
                      <div className="font-medium text-sm mb-2">{method.name}</div>
                      <ol className="space-y-1 list-decimal list-inside text-sm text-[var(--muted)]">
                        {method.steps.map((s, j) => <li key={j}>{s}</li>)}
                      </ol>
                      {method.final_answer_latex && (
                        <div className="mt-3 pt-2 border-t border-[var(--border)]">
                          <span className="text-xs text-[var(--muted)]">Result: </span>
                          <MathRenderer latex={method.final_answer_latex} />
                        </div>
                      )}
                    </div>
                  ))}
                  {result.alternative_methods.length === 0 && (
                    <p className="text-sm text-[var(--muted)] text-center py-8">No alternative methods available.</p>
                  )}
                </div>
              )}

              {activeTab === 'formulas' && (
                <div className="space-y-3">
                  {result.formulas_used.map((f, i) => (
                    <div key={i} className="flex items-start gap-3 p-3 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
                      <div className="flex-shrink-0 p-2 bg-primary-500/10 rounded-lg">
                        <MathRenderer latex={f.formula_latex} />
                      </div>
                      <div className="min-w-0">
                        <div className="font-medium text-sm">{f.name}</div>
                        {f.description && <div className="text-xs text-[var(--muted)] mt-0.5">{f.description}</div>}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Verify result */}
      {verifyResult && (
        <div className="space-y-4">
          <div className={`rounded-xl p-5 ${
            verifyResult.is_correct
              ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
              : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              {verifyResult.is_correct ? (
                <CheckCircle className="w-5 h-5 text-green-600" />
              ) : (
                <ArrowRight className="w-5 h-5 text-red-600" />
              )}
              <span className={`font-medium ${verifyResult.is_correct ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}`}>
                {verifyResult.is_correct ? 'Solution is correct!' : 'Solution needs corrections'}
              </span>
              <span className="text-sm ml-auto text-[var(--muted)]">
                Score: {(verifyResult.score * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-sm text-[var(--muted)]">{verifyResult.overall_feedback}</p>
          </div>

          {verifyResult.steps_analysis.length > 0 && (
            <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
              <div className="font-medium text-sm mb-3">Step Analysis</div>
              <div className="space-y-3">
                {verifyResult.steps_analysis.map((step, i) => (
                  <div key={i} className="flex gap-3 p-3 bg-[var(--bg)] rounded-lg">
                    <div className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center ${
                      step.is_correct ? 'bg-green-500/20 text-green-600' : 'bg-red-500/20 text-red-600'
                    }`}>
                      {step.is_correct ? '✓' : '✗'}
                    </div>
                    <div className="min-w-0 space-y-1">
                      <div className="text-sm font-medium">{step.student_step}</div>
                      <div className="text-xs text-[var(--muted)]">{step.feedback}</div>
                      {step.correction && (
                        <div className="text-xs text-primary-600 dark:text-primary-400">
                          Correction: {step.correction}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
