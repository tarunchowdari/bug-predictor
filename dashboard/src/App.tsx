import { BrowserRouter, Routes, Route } from 'react-router-dom';
import ProjectSelection from './pages/ProjectSelection';
import ProjectDetail from './pages/ProjectDetail';
import './index.css';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ProjectSelection />} />
        <Route path="/project/:repoName" element={<ProjectDetail />} />
      </Routes>
    </BrowserRouter>
  );
}
