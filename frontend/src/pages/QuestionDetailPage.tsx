import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { api } from '../services/api';
import { Question, Collection } from '../utils/types';
import MathRenderer from '../components/MathRenderer';
import {
  Star, Pin, Archive, Trash2, Copy, Edit, ArrowLeft, Plus, Minus,
  BookOpen, TrendingUp, Lightbulb, Tag, FolderOpen, CheckCircle, MoveRight, Clock
} from 'lucide-react';

export default function QuestionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [question, setQuestion] = useState<Question | null>(null);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [activeTab, setActiveTab] = useState<'steps' | 'alternatives' | 'formulas'>('steps');
  const [editMode, setEditMode] = useState(false);
  const [notes, setNotes] = useState('');
  const [tags, setTags] = useState('');
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (!id) return;
    api.getQuestion(id).then(setQuestion).catch(() => toast.error('Question not found'));
    api.getCollections().then(setCollections).catch(() => {});
  }, [id]);

  // Seed the notes/tags fields from the question whenever edit mode is
  // opened, instead of always starting blank (which meant saving silently
  // wiped out any existing notes/tags unless the user retyped them).
  const wasEditingRef = useRef(false);
  useEffect(() => {
    if (editMode && !wasEditingRef.current && question) {
      setNotes(question.notes || '');
      setTags((question.tags || []).join(', '));
    }
    wasEditingRef.current = editMode;
  }, [editMode, question]);

  const handleAction = async (action: string, fn: () => Promise<any>) => {
    try {
      const res = await fn();
      if (res && typeof res === 'object') setQuestion(prev => prev ? { ...prev, ...res } : prev);
      else {
        const updated = await api.getQuestion(id!);
        setQuestion(updated);
      }
      toast.success(`${action} updated`);
    } catch {
      toast.error(`${action} failed`);
    }
  };

  const handleSave = async () => {
    if (!id) return;
    try {
      const updated = await api.updateQuestion(id, {
        notes,
        tags: tags.split(',').map(t => t.trim()).filter(Boolean),
      });
      setQuestion(updated);
      setEditMode(false);
      toast.success('Saved');
    } catch {
      toast.error('Save failed');
    }
  };

  const handleMoveToCollection = async (collectionId: string) => {
    if (!id) return;
    try {
      await api.moveQuestion(collectionId, id);
      const updated = await api.getQuestion(id);
      setQuestion(updated);
      toast.success('Moved');
    } catch {
      toast.error('Move failed');
    }
  };

  if (!question) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-[var(--muted)]">Loading...</div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Back button */}
      <button
        onClick={() => navigate('/library')}
        className="flex items-center gap-2 text-sm text-[var(--muted)] hover:text-[var(--fg)] transition-colors"
      >
        <ArrowLeft className="w-4 h-4" /> Back to Library
      </button>

      {/* Header */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6">
        <div className="flex items-start justify-between flex-wrap gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-3">
              <span className="text-xs px-3 py-1 rounded-full font-medium bg-primary-500/10 text-primary-600 dark:text-primary-400">
                {question.topic || 'General'}
              </span>
              <span className={`text-xs px-3 py-1 rounded-full font-medium ${
                question.difficulty === 'Expert' ? 'bg-red-500/10 text-red-600 dark:text-red-400' :
                question.difficulty === 'Hard' ? 'bg-orange-500/10 text-orange-600 dark:text-orange-400' :
                question.difficulty === 'Medium' ? 'bg-yellow-500/10 text-yellow-600 dark:text-yellow-400' :
                'bg-green-500/10 text-green-600 dark:text-green-400'
              }`}>
                {question.difficulty || 'Unknown'}
              </span>
              {question.tags.map(t => (
                <span key={t} className="text-xs px-2 py-0.5 rounded-full bg-[var(--bg)] border border-[var(--border)] text-[var(--muted)]">
                  <Tag className="w-3 h-3 inline mr-1" />{t}
                </span>
              ))}
            </div>

            <h1 className="text-lg font-medium mb-2">{question.question}</h1>
            {question.question_latex && (
              <div className="p-3 bg-[var(--bg)] rounded-lg">
                <MathRenderer latex={question.question_latex} displayMode />
              </div>
            )}
          </div>

          {/* Action buttons */}
          <div className="flex gap-1">
            <button
              onClick={() => handleAction('Favorite', () => api.toggleFavorite(question.id))}
              className={`p-2 rounded-lg transition-colors ${question.is_favorite ? 'text-yellow-500 bg-yellow-500/10' : 'text-[var(--muted)] hover:text-yellow-500 hover:bg-yellow-500/5'}`}
              title="Favorite"
            >
              <Star className="w-4 h-4" fill={question.is_favorite ? 'currentColor' : 'none'} />
            </button>
            <button
              onClick={() => handleAction('Pin', () => api.togglePin(question.id))}
              className={`p-2 rounded-lg transition-colors ${question.is_pinned ? 'text-primary-500 bg-primary-500/10' : 'text-[var(--muted)] hover:text-primary-500 hover:bg-primary-500/5'}`}
              title="Pin"
            >
              <Pin className="w-4 h-4" />
            </button>
            <button
              onClick={() => handleAction('Archive', () => api.toggleArchive(question.id))}
              className={`p-2 rounded-lg transition-colors ${question.is_archived ? 'text-orange-500 bg-orange-500/10' : 'text-[var(--muted)] hover:text-orange-500 hover:bg-orange-500/5'}`}
              title="Archive"
            >
              <Archive className="w-4 h-4" />
            </button>
            <button
              onClick={async () => {
                try {
                  const dup = await api.duplicateQuestion(question.id);
                  toast.success('Duplicated');
                  navigate(`/questions/${dup.id}`);
                } catch { toast.error('Duplicate failed'); }
              }}
              className="p-2 rounded-lg text-[var(--muted)] hover:text-[var(--fg)] hover:bg-[var(--bg)] transition-colors"
              title="Duplicate"
            >
              <Copy className="w-4 h-4" />
            </button>
            <button
              onClick={() => setEditMode(!editMode)}
              className={`p-2 rounded-lg transition-colors ${editMode ? 'text-primary-500 bg-primary-500/10' : 'text-[var(--muted)] hover:text-[var(--fg)] hover:bg-[var(--bg)]'}`}
              title="Edit"
            >
              <Edit className="w-4 h-4" />
            </button>
            <button
              onClick={() => { if (confirm('Delete?')) { api.deleteQuestion(question.id).then(() => { navigate('/library'); toast.success('Deleted'); }); } }}
              className="p-2 rounded-lg text-[var(--muted)] hover:text-red-500 hover:bg-red-500/5 transition-colors"
              title="Delete"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex gap-4 mt-4 text-xs text-[var(--muted)]">
          <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {new Date(question.created_at).toLocaleString()}</span>
          <span>Views: {question.view_count}</span>
          <span>Confidence: {question.ai_confidence ? `${(question.ai_confidence * 100).toFixed(0)}%` : 'N/A'}</span>
        </div>
      </div>

      {/* Answer */}
      <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl p-6">
        <div className="flex items-center gap-2 text-green-700 dark:text-green-400 font-medium text-sm mb-3">
          <CheckCircle className="w-5 h-5" /> Final Answer
        </div>
        <MathRenderer latex={question.answer_latex || question.answer} displayMode />
      </div>

      {/* Tabs */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl overflow-hidden">
        <div className="flex border-b border-[var(--border)]">
          {[
            { key: 'steps', label: 'Step-by-Step', count: question.steps.length },
            { key: 'alternatives', label: 'Alternative Methods', count: question.alternative_methods.length },
            { key: 'formulas', label: 'Formulas Used', count: question.formulas_used.length },
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

        <div className="p-6 max-h-[600px] overflow-y-auto">
          {activeTab === 'steps' && (
            <div className="space-y-5">
              {question.steps.map((step, i) => (
                <div key={i} className="flex gap-4">
                  <button
                    onClick={() => {
                      setExpandedSteps(prev => {
                        const next = new Set(prev);
                        next.has(i) ? next.delete(i) : next.add(i);
                        return next;
                      });
                    }}
                    className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-500/10 flex items-center justify-center text-sm font-bold text-primary-600 dark:text-primary-400 hover:bg-primary-500/20 transition-colors"
                  >
                    {step.step_number}
                  </button>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm">{step.description}</div>
                    {step.expression_latex && (
                      <div className="mt-2 p-3 bg-[var(--bg)] rounded-lg">
                        <MathRenderer latex={step.expression_latex} displayMode />
                      </div>
                    )}
                    {step.justification && (
                      <div className="mt-2 flex items-start gap-2 text-xs text-[var(--muted)]">
                        <Lightbulb className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-yellow-500" />
                        <span>{step.justification}</span>
                      </div>
                    )}
                    {step.result && i === question.steps.length - 1 && (
                      <div className="mt-3 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
                        <MathRenderer latex={step.result_latex || step.result} displayMode />
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'alternatives' && (
            <div className="space-y-4">
              {question.alternative_methods.map((method, i) => (
                <div key={i} className="p-4 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
                  <div className="flex items-center gap-2 font-medium text-sm mb-3">
                    <MoveRight className="w-4 h-4 text-primary-500" />
                    {method.name}
                  </div>
                  <ol className="space-y-2 list-decimal list-inside text-sm text-[var(--muted)]">
                    {method.steps.map((s, j) => <li key={j}>{s}</li>)}
                  </ol>
                  {method.final_answer_latex && (
                    <div className="mt-3 pt-3 border-t border-[var(--border)]">
                      <span className="text-xs text-[var(--muted)]">Result: </span>
                      <MathRenderer latex={method.final_answer_latex} />
                    </div>
                  )}
                </div>
              ))}
              {question.alternative_methods.length === 0 && (
                <p className="text-sm text-[var(--muted)] text-center py-8">No alternative methods available.</p>
              )}
            </div>
          )}

          {activeTab === 'formulas' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {question.formulas_used.map((f, i) => (
                <div key={i} className="flex items-start gap-3 p-4 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
                  <div className="flex-shrink-0 p-2 bg-primary-500/10 rounded-lg">
                    <MathRenderer latex={f.formula_latex} />
                  </div>
                  <div className="min-w-0">
                    <div className="font-medium text-sm">{f.name}</div>
                    {f.description && (
                      <div className="text-xs text-[var(--muted)] mt-1">{f.description}</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Edit section */}
      {editMode && (
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6 space-y-4">
          <h3 className="font-medium text-sm">Edit Question</h3>
          <div>
            <label className="block text-xs text-[var(--muted)] mb-1">Notes</label>
            <textarea
              value={notes}
              onChange={e => setNotes(e.target.value)}
              rows={3}
              className="w-full px-3 py-2 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="Add notes... (will be set on save)"
            />
          </div>
          <div>
            <label className="block text-xs text-[var(--muted)] mb-1">Tags (comma-separated)</label>
            <input
              value={tags}
              onChange={e => setTags(e.target.value)}
              className="w-full px-3 py-2 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="limits, derivatives, etc."
            />
          </div>
          <div className="flex gap-2">
            <button onClick={handleSave} className="px-4 py-2 bg-primary-500 text-white rounded-lg text-sm hover:bg-primary-600">Save</button>
            <button onClick={() => setEditMode(false)} className="px-4 py-2 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-sm">Cancel</button>
          </div>

          {/* Move to collection */}
          <div className="border-t border-[var(--border)] pt-4">
            <div className="flex items-center gap-2 text-sm text-[var(--muted)] mb-2">
              <FolderOpen className="w-4 h-4" /> Move to Collection
            </div>
            <div className="flex flex-wrap gap-2">
              {collections.map(c => (
                <button
                  key={c.id}
                  onClick={() => handleMoveToCollection(c.id)}
                  className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
                    question.collection_id === c.id
                      ? 'border-primary-500 bg-primary-500/10 text-primary-600'
                      : 'border-[var(--border)] hover:border-primary-500/30'
                  }`}
                >
                  {c.name} ({c.question_count})
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Metadata */}
      <div className="text-xs text-[var(--muted)] text-center">
        Created: {new Date(question.created_at).toLocaleString()} •
        Updated: {new Date(question.updated_at).toLocaleString()} •
        ID: {question.id}
      </div>
    </div>
  );
}
