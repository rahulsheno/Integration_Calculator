import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { AppProvider } from './context/AppContext';
import Layout from './components/Layout';
import SolverPage from './pages/SolverPage';
import LibraryPage from './pages/LibraryPage';
import CollectionsPage from './pages/CollectionsPage';
import QuestionDetailPage from './pages/QuestionDetailPage';

export default function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: 'var(--card)',
              color: 'var(--fg)',
              border: '1px solid var(--border)',
              fontSize: '13px',
            },
          }}
        />
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<SolverPage />} />
            <Route path="/solve" element={<SolverPage />} />
            <Route path="/library" element={<LibraryPage />} />
            <Route path="/collections" element={<CollectionsPage />} />
            <Route path="/questions/:id" element={<QuestionDetailPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppProvider>
  );
}
