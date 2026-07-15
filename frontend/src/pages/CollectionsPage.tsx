import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { api } from '../services/api';
import { Collection } from '../utils/types';
import {
  FolderOpen, Plus, Trash2, Edit, X, ChevronRight, Folder,
  Hash, Palette
} from 'lucide-react';

const COLORS = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];
const ICONS = ['📁', '📊', '🧮', '📝', '📚', '⭐', '🔥', '💡', '🎯', '📐'];

export default function CollectionsPage() {
  const navigate = useNavigate();
  const [collections, setCollections] = useState<Collection[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [editId, setEditId] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [parentId, setParentId] = useState<string | null>(null);
  const [colorLabel, setColorLabel] = useState('');
  const [icon, setIcon] = useState('');

  useEffect(() => {
    loadCollections();
  }, []);

  const loadCollections = async () => {
    try {
      const res = await api.getCollections();
      setCollections(res);
    } catch { toast.error('Failed to load collections'); }
  };

  const handleCreate = async () => {
    if (!name.trim()) return;
    try {
      await api.createCollection({ name: name.trim(), description, parent_id: parentId, color_label: colorLabel, icon });
      toast.success('Collection created');
      setShowCreate(false);
      resetForm();
      loadCollections();
    } catch { toast.error('Create failed'); }
  };

  const handleUpdate = async (id: string) => {
    if (!name.trim()) return;
    try {
      await api.updateCollection(id, { name: name.trim(), description, parent_id: parentId, color_label: colorLabel, icon });
      toast.success('Updated');
      setEditId(null);
      resetForm();
      loadCollections();
    } catch { toast.error('Update failed'); }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Delete "${name}"? Questions will be unlinked.`)) return;
    try {
      await api.deleteCollection(id);
      toast.success('Deleted');
      loadCollections();
    } catch { toast.error('Delete failed'); }
  };

  const resetForm = () => {
    setName('');
    setDescription('');
    setParentId(null);
    setColorLabel('');
    setIcon('');
  };

  const startEdit = (c: Collection) => {
    setEditId(c.id);
    setName(c.name);
    setDescription(c.description || '');
    setParentId(c.parent_id);
    setColorLabel(c.color_label || '');
    setIcon(c.icon || '');
  };

  const rootCollections = collections.filter(c => !c.parent_id);
  const getChildren = (parentId: string) => collections.filter(c => c.parent_id === parentId);

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">Collections</h2>
        <button
          onClick={() => { setShowCreate(true); setEditId(null); resetForm(); }}
          className="flex items-center gap-2 px-4 py-2 bg-primary-500 text-white rounded-lg text-sm hover:bg-primary-600 transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Collection
        </button>
      </div>

      {/* Create/Edit form */}
      {(showCreate || editId) && (
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6 space-y-4">
          <h3 className="font-medium text-sm">{editId ? 'Edit Collection' : 'Create Collection'}</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-[var(--muted)] mb-1">Name</label>
              <input
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder="e.g., Integration Problems"
                className="w-full px-3 py-2 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
            <div>
              <label className="block text-xs text-[var(--muted)] mb-1">Description</label>
              <input
                value={description}
                onChange={e => setDescription(e.target.value)}
                placeholder="Optional description"
                className="w-full px-3 py-2 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
            <div>
              <label className="block text-xs text-[var(--muted)] mb-1">Parent Collection</label>
              <select
                value={parentId || ''}
                onChange={e => setParentId(e.target.value || null)}
                className="w-full px-3 py-2 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="">None (Root)</option>
                {collections.map(c => (
                  <option key={c.id} value={c.id} disabled={c.id === editId}>
                    {c.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-[var(--muted)] mb-1">Color Label</label>
              <div className="flex gap-2 flex-wrap">
                {COLORS.map(c => (
                  <button
                    key={c}
                    onClick={() => setColorLabel(c === colorLabel ? '' : c)}
                    className={`w-6 h-6 rounded-full transition-transform ${colorLabel === c ? 'scale-125 ring-2 ring-offset-1 ring-offset-[var(--card)]' : ''}`}
                    style={{ backgroundColor: c }}
                  />
                ))}
              </div>
            </div>
          </div>
          <div>
            <label className="block text-xs text-[var(--muted)] mb-1">Icon</label>
            <div className="flex gap-2 flex-wrap">
              {ICONS.map(em => (
                <button
                  key={em}
                  onClick={() => setIcon(em === icon ? '' : em)}
                  className={`w-8 h-8 flex items-center justify-center rounded-lg text-lg transition-colors ${icon === em ? 'bg-primary-500/20 ring-2 ring-primary-500' : 'bg-[var(--bg)] hover:bg-[var(--border)]'}`}
                >
                  {em}
                </button>
              ))}
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => editId ? handleUpdate(editId) : handleCreate()}
              className="px-4 py-2 bg-primary-500 text-white rounded-lg text-sm hover:bg-primary-600"
            >
              {editId ? 'Save Changes' : 'Create'}
            </button>
            <button
              onClick={() => { setShowCreate(false); setEditId(null); resetForm(); }}
              className="px-4 py-2 bg-[var(--bg)] border border-[var(--border)] rounded-lg text-sm"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Collections list */}
      <div className="space-y-3">
        {rootCollections.length === 0 ? (
          <div className="text-center py-12 bg-[var(--card)] border border-[var(--border)] rounded-xl">
            <FolderOpen className="w-12 h-12 mx-auto mb-3 text-[var(--muted)]" />
            <p className="text-[var(--muted)] text-sm">No collections yet. Create one to organize your questions.</p>
          </div>
        ) : (
          rootCollections.map(c => (
            <CollectionItem
              key={c.id}
              collection={c}
              children_={getChildren(c.id)}
              getChildren={getChildren}
              onEdit={startEdit}
              onDelete={handleDelete}
              onClick={(id: string) => navigate(`/library?collection_id=${id}`)}
              depth={0}
            />
          ))
        )}
      </div>
    </div>
  );
}

function CollectionItem({
  collection,
  children_,
  getChildren,
  onEdit,
  onDelete,
  onClick,
  depth,
}: {
  collection: Collection;
  children_: Collection[];
  getChildren: (parentId: string) => Collection[];
  onEdit: (c: Collection) => void;
  onDelete: (id: string, name: string) => void;
  onClick: (id: string) => void;
  depth: number;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div>
      <div
        className="group flex items-center gap-3 p-4 bg-[var(--card)] border border-[var(--border)] rounded-xl hover:border-primary-500/30 transition-all cursor-pointer"
        style={{ marginLeft: depth * 16 }}
      >
        {/* Drag handle */}
        <div className="flex-shrink-0 cursor-grab text-[var(--border)] hover:text-[var(--muted)]">
          <Hash className="w-4 h-4" />
        </div>

        {/* Icon & color */}
        <div
          className="flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center text-xl"
          style={{ backgroundColor: collection.color_label ? `${collection.color_label}20` : 'var(--bg)' }}
        >
          {collection.icon || '📁'}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0" onClick={() => onClick(collection.id)}>
          <div className="font-medium text-sm">{collection.name}</div>
          {collection.description && (
            <div className="text-xs text-[var(--muted)] mt-0.5">{collection.description}</div>
          )}
          <div className="text-xs text-[var(--muted)] mt-1">{collection.question_count} questions</div>
        </div>

        {/* Actions */}
        <div className="flex-shrink-0 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={e => { e.stopPropagation(); onEdit(collection); }}
            className="p-1.5 rounded-md hover:bg-[var(--bg)] text-[var(--muted)] hover:text-[var(--fg)]"
          >
            <Edit className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={e => { e.stopPropagation(); onDelete(collection.id, collection.name); }}
            className="p-1.5 rounded-md hover:bg-red-50 text-[var(--muted)] hover:text-red-500"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Expand/collapse if has children */}
        {children_.length > 0 && (
          <button
            onClick={e => { e.stopPropagation(); setExpanded(e => !e); }}
            className="flex-shrink-0 p-1 rounded-md hover:bg-[var(--bg)]"
          >
            <ChevronRight className={`w-4 h-4 transition-transform ${expanded ? 'rotate-90' : ''}`} />
          </button>
        )}
      </div>

      {/* Children */}
      {expanded && children_.map(child => (
        <CollectionItem
          key={child.id}
          collection={child}
          children_={getChildren(child.id)}
          getChildren={getChildren}
          onEdit={onEdit}
          onDelete={onDelete}
          onClick={onClick}
          depth={depth + 1}
        />
      ))}
    </div>
  );
}
