const BASE_URL = '/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

export const api = {
  solve: (data: { expression: string; topic_hint?: string | null; include_graph?: boolean; session_id?: string | null }) =>
    request<any>('/solve', { method: 'POST', body: JSON.stringify({ ...data, include_graph: data.include_graph ?? true }) }),

  verify: (data: { expression: string; student_solution: string }) =>
    request<any>('/solve/verify', { method: 'POST', body: JSON.stringify(data) }),

  upload: (file: File) => {
    const fd = new FormData();
    fd.append('file', file);
    return fetch(`${BASE_URL}/upload`, { method: 'POST', body: fd }).then(async r => {
      const body = await r.json().catch(() => ({ detail: r.statusText }));
      if (!r.ok) {
        throw new Error(body.detail || 'Upload failed');
      }
      return body;
    });
  },

  getQuestions: (params?: Record<string, string>) => {
    const qs = params ? '?' + new URLSearchParams(params).toString() : '';
    return request<any>(`/questions${qs}`);
  },

  getQuestion: (id: string) => request<any>(`/questions/${id}`),

  createQuestion: (data: any) => request<any>('/questions', { method: 'POST', body: JSON.stringify(data) }),

  updateQuestion: (id: string, data: any) => request<any>(`/questions/${id}`, { method: 'PUT', body: JSON.stringify(data) }),

  deleteQuestion: (id: string) => request<any>(`/questions/${id}`, { method: 'DELETE' }),

  duplicateQuestion: (id: string) => request<any>(`/questions/${id}/duplicate`, { method: 'POST' }),

  toggleFavorite: (id: string) => request<any>(`/questions/${id}/favorite`, { method: 'POST' }),

  togglePin: (id: string) => request<any>(`/questions/${id}/pin`, { method: 'POST' }),

  toggleArchive: (id: string) => request<any>(`/questions/${id}/archive`, { method: 'POST' }),

  getStats: () => request<any>('/questions/stats'),

  getCollections: () => request<any[]>('/collections'),

  createCollection: (data: any) => request<any>('/collections', { method: 'POST', body: JSON.stringify(data) }),

  updateCollection: (id: string, data: any) => request<any>(`/collections/${id}`, { method: 'PUT', body: JSON.stringify(data) }),

  deleteCollection: (id: string) => request<any>(`/collections/${id}`, { method: 'DELETE' }),

  moveQuestion: (collectionId: string, questionId: string) =>
    request<any>(`/collections/${collectionId}/move/${questionId}`, { method: 'POST' }),

  bulkDelete: (ids: string[]) => request<any>('/questions/bulk/delete', { method: 'POST', body: JSON.stringify({ question_ids: ids }) }),

  bulkArchive: (ids: string[]) => request<any>('/questions/bulk/archive', { method: 'POST', body: JSON.stringify({ question_ids: ids }) }),

  bulkTag: (ids: string[], tags: string[]) => request<any>('/questions/bulk/tag', { method: 'POST', body: JSON.stringify({ question_ids: ids, tags }) }),

  importQuestions: (questions: any[], collectionId?: string | null) =>
    request<any>('/questions/import', {
      method: 'POST',
      body: JSON.stringify({ collection_id: collectionId ?? null, questions }),
    }),

  exportQuestions: (ids?: string[], format?: string) => {
    const qs = new URLSearchParams({ format: format || 'json' });
    if (ids?.length) qs.set('question_ids', ids.join(','));
    return fetch(`${BASE_URL}/questions/export/format?${qs.toString()}`).then(r => r.blob());
  },
};
