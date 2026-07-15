import { useState, useEffect, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { api } from '../services/api';
import { Question, QuestionStats, Collection } from '../utils/types';
import MathRenderer from '../components/MathRenderer';
import {
  Search, Filter, Star, Pin, Archive, Trash2, Copy, Plus, Grid3X3,
  List, Table2, Download, Upload, Tag, FolderOpen, BookOpen,
  TrendingUp, LayoutList, CheckSquare, Square, Clock, Eye, MoreHorizontal,
  X, ChevronLeft, ChevronRight, SortAsc
} from 'lucide-react';

type ViewMode = 'list' | 'grid' | 'table';
type SortBy = 'newest' | 'oldest' | 'most_viewed' | 'alphabetical';

export default function LibraryPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [questions, setQuestions] = useState<Question[]>([]);
  const [stats, setStats] = useState<QuestionStats | null>(null);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [sortBy, setSortBy] = useState<SortBy>('newest');
  const [search, setSearch] = useState('');
  const [selectedTopic, setSelectedTopic] = useState('');
  const [selectedDifficulty, setSelectedDifficulty] = useState('');
  const [selectedTags, setSelectedTags] = useState('');
  const [selectedCollection, setSelectedCollection] = useState(searchParams.get('collection_id') || '');
  const [showArchived, setShowArchived] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [showFilters, setShowFilters] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const [showImport, setShowImport] = useState(false);
  const [bulkTagInput, setBulkTagInput] = useState('');
  const [showBulkTag, setShowBulkTag] = useState(false);

  const topics = [
    'Limits', 'Continuity', 'Differentiation', 'Implicit Differentiation',
    'Higher-Order Derivatives', 'Partial Derivatives', 'Chain Rule',
    'Product & Quotient Rules', 'Optimization', 'Related Rates',
    'Indefinite Integrals', 'Definite Integrals', 'Integration by Parts',
    'U-Substitution', 'Partial Fractions', 'Trigonometric Integrals',
    'Trigonometric Substitution', 'Improper Integrals', 'Multiple Integrals',
    'Vector Calculus', 'Line Integrals', 'Surface Integrals',
    "Green's Theorem", "Stokes' Theorem", 'Divergence Theorem',
    'Differential Equations', 'Taylor & Maclaurin Series', 'Fourier Series',
    'Laplace Transforms', 'Polar Coordinates', 'Parametric Equations',
  ];

  const fetchQuestions = useCallback(async () => {
    setLoading(true);
    try {
      const params: Record<string, string> = {
        page: String(page),
        page_size: '20',
        sort_by: sortBy,
        archived: String(showArchived),
      };
      if (search) params.search = search;
      if (selectedTopic) params.topic = selectedTopic;
      if (selectedDifficulty) params.difficulty = selectedDifficulty;
      if (selectedTags) params.tags = selectedTags;
      if (selectedCollection) params.collection_id = selectedCollection;

      const res = await api.getQuestions(params);
      setQuestions(res.items);
      setTotalPages(res.total_pages);
    } catch (e: any) {
      toast.error('Failed to load questions');
    } finally {
      setLoading(false);
    }
  }, [page, sortBy, search, selectedTopic, selectedDifficulty, selectedTags, selectedCollection, showArchived]);

  useEffect(() => { fetchQuestions(); }, [fetchQuestions]);
  useEffect(() => { setPage(1); }, [search, selectedTopic, selectedDifficulty, selectedTags, selectedCollection, showArchived, sortBy]);

  // Pick up collection_id whenever it's present in the URL (e.g. arriving
  // from the Collections page), including when Library is already mounted
  // and the user clicks a different collection.
  useEffect(() => {
    const fromUrl = searchParams.get('collection_id') || '';
    setSelectedCollection(fromUrl);
  }, [searchParams]);

  useEffect(() => {
    api.getStats().then(setStats).catch(() => {});
    api.getCollections().then(setCollections).catch(() => {});
  }, []);

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === questions.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(questions.map(q => q.id)));
    }
  };

  const bulkAction = async (action: string, fn: () => Promise<any>) => {
    if (selectedIds.size === 0) return toast.error('Select questions first');
    try {
      await fn();
      toast.success(`${action} complete`);
      setSelectedIds(new Set());
      fetchQuestions();
    } catch {
      toast.error(`${action} failed`);
    }
  };

  const handleExport = async (format: string) => {
    try {
      const ids = selectedIds.size > 0 ? Array.from(selectedIds) : undefined;
      const blob = await api.exportQuestions(ids, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `questions.${format}`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success('Exported');
    } catch {
      toast.error('Export failed');
    }
  };

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      let data: Record<string, any>[] = [];
      if (file.name.endsWith('.json')) {
        const parsed = JSON.parse(text);
        data = Array.isArray(parsed) ? parsed : (parsed.questions || []);
      } else if (file.name.endsWith('.csv')) {
        const lines = text.split('\n').filter(l => l.trim().length > 0);
        const headers = lines[0].split(',').map(h => h.trim());
        data = lines.slice(1).map(line => {
          const vals = line.split(',');
          return Object.fromEntries(headers.map((h, i) => [h, (vals[i] || '').trim()]));
        });
      } else {
        return toast.error('Unsupported format');
      }
      if (data.length === 0) {
        return toast.error('No questions found in file');
      }
      // Ensure every record has the fields the backend requires, and turn
      // the CSV's comma-joined tags string back into an array.
      const questions = data.map(item => ({
        question: item.question || '',
        answer: item.answer || '',
        question_latex: item.question_latex || null,
        answer_latex: item.answer_latex || null,
        topic: item.topic || null,
        difficulty: item.difficulty || null,
        notes: item.notes || null,
        tags: Array.isArray(item.tags)
          ? item.tags
          : typeof item.tags === 'string' && item.tags
            ? item.tags.split(',').map((t: string) => t.trim()).filter(Boolean)
            : [],
      }));
      await api.importQuestions(questions, selectedCollection || null);
      toast.success(`Imported ${questions.length} question${questions.length === 1 ? '' : 's'}`);
      fetchQuestions();
    } catch {
      toast.error('Import failed');
    }
    setShowImport(false);
  };

  const clearFilters = () => {
    setSearch('');
    setSelectedTopic('');
    setSelectedDifficulty('');
    setSelectedTags('');
    setSelectedCollection('');
    setShowArchived(false);
  };

  const hasFilters = search || selectedTopic || selectedDifficulty || selectedTags || selectedCollection || showArchived;

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Toolbar */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 space-y-4">
        <div className="flex items-center gap-2 flex-wrap">
          {/* Search */}
          <div className="flex-1 min-w-[200px] relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--muted)]" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search questions..."
              className="w-full pl-10 pr-4 py-2 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>

          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm border transition-colors ${
              showFilters || hasFilters
                ? 'border-primary-500 bg-primary-500/10 text-primary-600'
                : 'border-[var(--border)] text-[var(--muted)] hover:text-[var(--fg)]'
            }`}
          >
            <Filter className="w-4 h-4" />
            Filters
          </button>

          <button
            onClick={() => setShowStats(!showStats)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm border transition-colors ${
              showStats ? 'border-primary-500 bg-primary-500/10 text-primary-600' : 'border-[var(--border)] text-[var(--muted)] hover:text-[var(--fg)]'
            }`}
          >
            <TrendingUp className="w-4 h-4" />
            Stats
          </button>

          <button
            onClick={() => setShowImport(!showImport)}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm border border-[var(--border)] text-[var(--muted)] hover:text-[var(--fg)] transition-colors"
          >
            <Upload className="w-4 h-4" />
            Import
          </button>

          <button
            onClick={() => handleExport('json')}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm border border-[var(--border)] text-[var(--muted)] hover:text-[var(--fg)] transition-colors"
          >
            <Download className="w-4 h-4" />
            Export
          </button>

          <button
            onClick={() => navigate('/solve')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm bg-primary-500 text-white hover:bg-primary-600 transition-colors"
          >
            <Plus className="w-4 h-4" />
            Solve New
          </button>
        </div>

        {/* View mode + sort */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-1 bg-[var(--bg)] border border-[var(--border)] rounded-lg p-1">
            {([
              { key: 'grid', icon: Grid3X3 },
              { key: 'list', icon: List },
              { key: 'table', icon: Table2 },
            ] as const).map(opt => (
              <button
                key={opt.key}
                onClick={() => setViewMode(opt.key)}
                className={`p-1.5 rounded-md transition-colors ${
                  viewMode === opt.key ? 'bg-primary-500 text-white' : 'text-[var(--muted)] hover:text-[var(--fg)]'
                }`}
              >
                <opt.icon className="w-4 h-4" />
              </button>
            ))}
          </div>

          <select
            value={sortBy}
            onChange={e => setSortBy(e.target.value as SortBy)}
            className="px-3 py-1.5 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="newest">Newest</option>
            <option value="oldest">Oldest</option>
            <option value="most_viewed">Most Viewed</option>
            <option value="alphabetical">Alphabetical</option>
          </select>

          {selectedIds.size > 0 && (
            <div className="flex items-center gap-1 ml-auto">
              <span className="text-xs text-[var(--muted)]">{selectedIds.size} selected</span>
              <button
                onClick={() => setShowBulkTag(!showBulkTag)}
                className="px-2 py-1 text-xs rounded-md border border-[var(--border)] hover:border-primary-500 transition-colors"
              >
                <Tag className="w-3 h-3 inline mr-1" />Tag
              </button>
              <button
                onClick={() => bulkAction('Archived', () => api.bulkArchive(Array.from(selectedIds)))}
                className="px-2 py-1 text-xs rounded-md border border-[var(--border)] hover:border-orange-500 transition-colors text-orange-600"
              >
                <Archive className="w-3 h-3 inline mr-1" />Archive
              </button>
              <button
                onClick={() => { if (confirm(`Delete ${selectedIds.size} questions?`)) bulkAction('Deleted', () => api.bulkDelete(Array.from(selectedIds))); }}
                className="px-2 py-1 text-xs rounded-md border border-[var(--border)] hover:border-red-500 transition-colors text-red-600"
              >
                <Trash2 className="w-3 h-3 inline mr-1" />Delete
              </button>
            </div>
          )}
        </div>

        {/* Bulk tag input */}
        {showBulkTag && (
          <div className="flex gap-2 items-center">
            <input
              value={bulkTagInput}
              onChange={e => setBulkTagInput(e.target.value)}
              placeholder="Tags (comma-separated)"
              className="flex-1 px-3 py-1.5 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <button
              onClick={() => {
                const tags = bulkTagInput.split(',').map(t => t.trim()).filter(Boolean);
                bulkAction('Tagged', () => api.bulkTag(Array.from(selectedIds), tags));
                setShowBulkTag(false);
                setBulkTagInput('');
              }}
              className="px-3 py-1.5 bg-primary-500 text-white rounded-lg text-sm"
            >
              Apply
            </button>
          </div>
        )}

        {/* Filters panel */}
        {showFilters && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 p-4 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
            <select
              value={selectedTopic}
              onChange={e => setSelectedTopic(e.target.value)}
              className="px-3 py-2 bg-[var(--card)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="">All Topics</option>
              {topics.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            <select
              value={selectedDifficulty}
              onChange={e => setSelectedDifficulty(e.target.value)}
              className="px-3 py-2 bg-[var(--card)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="">All Difficulty</option>
              <option value="Easy">Easy</option>
              <option value="Medium">Medium</option>
              <option value="Hard">Hard</option>
              <option value="Expert">Expert</option>
            </select>
            <input
              value={selectedTags}
              onChange={e => setSelectedTags(e.target.value)}
              placeholder="Tags (comma-separated)"
              className="px-3 py-2 bg-[var(--card)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <select
              value={selectedCollection}
              onChange={e => setSelectedCollection(e.target.value)}
              className="px-3 py-2 bg-[var(--card)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="">All Collections</option>
              {collections.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={showArchived}
                onChange={() => setShowArchived(!showArchived)}
                className="rounded border-[var(--border)]"
              />
              Show Archived
            </label>
            {hasFilters && (
              <button onClick={clearFilters} className="text-xs text-primary-500 hover:underline col-span-2 md:col-span-1">
                Clear all filters
              </button>
            )}
          </div>
        )}

        {/* Import panel */}
        {showImport && (
          <div className="p-4 bg-[var(--bg)] rounded-lg border border-[var(--border)]">
            <div className="flex items-center gap-2 text-sm text-[var(--muted)] mb-2">
              <Upload className="w-4 h-4" /> Upload JSON or CSV
            </div>
            <input
              type="file"
              accept=".json,.csv"
              onChange={handleImport}
              className="text-sm"
            />
          </div>
        )}
      </div>

      {/* Stats panel */}
      {showStats && stats && (
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6 space-y-4">
          <h3 className="font-medium">Library Statistics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-center">
              <div className="text-2xl font-bold text-primary-500">{stats.total_questions}</div>
              <div className="text-xs text-[var(--muted)] mt-1">Total Questions</div>
            </div>
            {['Easy', 'Medium', 'Hard', 'Expert'].map(d => (
              <div key={d} className="p-4 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-center">
                <div className="text-2xl font-bold text-primary-500">{stats.difficulty_distribution?.[d] || 0}</div>
                <div className="text-xs text-[var(--muted)] mt-1">{d}</div>
              </div>
            ))}
          </div>
          {stats.recently_added.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">Recently Added</div>
              <div className="space-y-2">
                {stats.recently_added.slice(0, 3).map(q => (
                  <div key={q.id}
                    onClick={() => navigate(`/questions/${q.id}`)}
                    className="p-3 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-sm hover:border-primary-500/30 cursor-pointer transition-colors"
                  >
                    {q.question.substring(0, 100)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Questions display */}
      <div className="space-y-4">
        {loading ? (
          <div className="text-center py-12 text-[var(--muted)]">Loading questions...</div>
        ) : questions.length === 0 ? (
          <div className="text-center py-12 bg-[var(--card)] border border-[var(--border)] rounded-xl">
            <BookOpen className="w-12 h-12 mx-auto mb-3 text-[var(--muted)]" />
            <p className="text-[var(--muted)]">No questions found.</p>
            <button onClick={() => navigate('/solve')} className="mt-3 px-4 py-2 bg-primary-500 text-white rounded-lg text-sm hover:bg-primary-600">
              Solve a Problem
            </button>
          </div>
        ) : viewMode === 'grid' ? (
          /* Grid View */
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {questions.map(q => (
              <div
                key={q.id}
                onClick={() => navigate(`/questions/${q.id}`)}
                className="group bg-[var(--card)] border border-[var(--border)] hover:border-primary-500/40 rounded-xl p-4 cursor-pointer transition-all hover:shadow-sm"
              >
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex items-center gap-1 flex-wrap">
                    <span className="text-xs px-2 py-0.5 rounded-full bg-primary-500/10 text-primary-600 dark:text-primary-400">
                      {q.topic || 'General'}
                    </span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      q.difficulty === 'Expert' ? 'bg-red-500/10 text-red-600' :
                      q.difficulty === 'Hard' ? 'bg-orange-500/10 text-orange-600' :
                      q.difficulty === 'Medium' ? 'bg-yellow-500/10 text-yellow-600' :
                      'bg-green-500/10 text-green-600'
                    }`}>
                      {q.difficulty || 'N/A'}
                    </span>
                  </div>
                  <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    {q.is_pinned && <Pin className="w-3 h-3 text-primary-500" />}
                    {q.is_favorite && <Star className="w-3 h-3 text-yellow-500" fill="currentColor" />}
                    {q.is_archived && <Archive className="w-3 h-3 text-orange-500" />}
                  </div>
                </div>

                <p className="text-sm line-clamp-2 mb-2">{q.question}</p>

                {q.question_latex && (
                  <div className="mb-2 text-xs overflow-hidden">
                    <MathRenderer latex={q.question_latex} />
                  </div>
                )}

                {q.answer && (
                  <div className="p-2 bg-green-50 dark:bg-green-900/10 rounded-md text-xs">
                    <MathRenderer latex={q.answer_latex || q.answer} />
                  </div>
                )}

                <div className="flex items-center gap-3 mt-3 text-xs text-[var(--muted)]">
                  <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{new Date(q.created_at).toLocaleDateString()}</span>
                  <span className="flex items-center gap-1"><Eye className="w-3 h-3" />{q.view_count}</span>
                  {q.collection_name && (
                    <span className="flex items-center gap-1"><FolderOpen className="w-3 h-3" />{q.collection_name}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : viewMode === 'list' ? (
          /* List View */
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl divide-y divide-[var(--border)]">
            {questions.map(q => (
              <div
                key={q.id}
                onClick={() => navigate(`/questions/${q.id}`)}
                className="flex items-center gap-4 p-4 hover:bg-[var(--bg)] cursor-pointer transition-colors group"
              >
                <div
                  onClick={e => { e.stopPropagation(); toggleSelect(q.id); }}
                  className="flex-shrink-0"
                >
                  {selectedIds.has(q.id) ? (
                    <CheckSquare className="w-5 h-5 text-primary-500" />
                  ) : (
                    <Square className="w-5 h-5 text-[var(--border)]" />
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs px-2 py-0.5 rounded-full bg-primary-500/10 text-primary-600">
                      {q.topic || 'General'}
                    </span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      q.difficulty === 'Expert' ? 'bg-red-500/10 text-red-600' :
                      q.difficulty === 'Hard' ? 'bg-orange-500/10 text-orange-600' :
                      q.difficulty === 'Medium' ? 'bg-yellow-500/10 text-yellow-600' :
                      'bg-green-500/10 text-green-600'
                    }`}>
                      {q.difficulty}
                    </span>
                    {q.is_pinned && <Pin className="w-3 h-3 text-primary-500" />}
                    {q.is_favorite && <Star className="w-3 h-3 text-yellow-500" fill="currentColor" />}
                    {q.is_archived && <Archive className="w-3 h-3 text-orange-500" />}
                  </div>
                  <p className="text-sm truncate">{q.question}</p>
                  <div className="text-xs text-[var(--muted)] truncate mt-0.5">
                    {q.answer?.substring(0, 80)}
                  </div>
                </div>

                <div className="flex-shrink-0 text-xs text-[var(--muted)] flex items-center gap-3">
                  <span>{new Date(q.created_at).toLocaleDateString()}</span>
                  <span className="flex items-center gap-1"><Eye className="w-3 h-3" />{q.view_count}</span>
                  {q.collection_name && (
                    <span className="flex items-center gap-1"><FolderOpen className="w-3 h-3" />{q.collection_name}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          /* Table View */
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl overflow-hidden overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--border)] bg-[var(--bg)]">
                  <th className="p-3 text-left w-10">
                    <button onClick={toggleSelectAll}>
                      {selectedIds.size === questions.length && questions.length > 0 ? (
                        <CheckSquare className="w-4 h-4 text-primary-500" />
                      ) : (
                        <Square className="w-4 h-4 text-[var(--border)]" />
                      )}
                    </button>
                  </th>
                  <th className="p-3 text-left font-medium text-[var(--muted)]">Question</th>
                  <th className="p-3 text-left font-medium text-[var(--muted)]">Topic</th>
                  <th className="p-3 text-left font-medium text-[var(--muted)]">Difficulty</th>
                  <th className="p-3 text-left font-medium text-[var(--muted)]">Answer</th>
                  <th className="p-3 text-left font-medium text-[var(--muted)]">Date</th>
                  <th className="p-3 text-left font-medium text-[var(--muted)]">Views</th>
                  <th className="p-3 text-left font-medium text-[var(--muted)]">Collection</th>
                </tr>
              </thead>
              <tbody>
                {questions.map(q => (
                  <tr
                    key={q.id}
                    onClick={() => navigate(`/questions/${q.id}`)}
                    className="border-b border-[var(--border)] hover:bg-[var(--bg)] cursor-pointer transition-colors"
                  >
                    <td className="p-3">
                      <button onClick={e => { e.stopPropagation(); toggleSelect(q.id); }}>
                        {selectedIds.has(q.id) ? (
                          <CheckSquare className="w-4 h-4 text-primary-500" />
                        ) : (
                          <Square className="w-4 h-4 text-[var(--border)]" />
                        )}
                      </button>
                    </td>
                    <td className="p-3 max-w-[200px] truncate">{q.question}</td>
                    <td className="p-3">
                      <span className="text-xs px-2 py-0.5 rounded-full bg-primary-500/10 text-primary-600">
                        {q.topic || 'General'}
                      </span>
                    </td>
                    <td className="p-3">
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        q.difficulty === 'Expert' ? 'bg-red-500/10 text-red-600' :
                        q.difficulty === 'Hard' ? 'bg-orange-500/10 text-orange-600' :
                        q.difficulty === 'Medium' ? 'bg-yellow-500/10 text-yellow-600' :
                        'bg-green-500/10 text-green-600'
                      }`}>
                        {q.difficulty || 'N/A'}
                      </span>
                    </td>
                    <td className="p-3 max-w-[150px] truncate text-xs text-[var(--muted)]">{q.answer || '-'}</td>
                    <td className="p-3 text-xs text-[var(--muted)]">{new Date(q.created_at).toLocaleDateString()}</td>
                    <td className="p-3 text-xs text-[var(--muted)]">{q.view_count}</td>
                    <td className="p-3 text-xs text-[var(--muted)]">{q.collection_name || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="p-2 rounded-lg border border-[var(--border)] disabled:opacity-30"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-sm text-[var(--muted)]">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="p-2 rounded-lg border border-[var(--border)] disabled:opacity-30"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
