// src/App.tsx
import { useState } from 'react';
import UploadScreen from './components/UploadScreen';
import JobStatus from './components/JobStatus';

function App() {
  const [jobId, setJobId] = useState<string | null>(null);

  return (
    <>
      {!jobId ? (
        <UploadScreen onJobCreated={setJobId} />
      ) : (
        <JobStatus jobId={jobId} onReset={() => setJobId(null)} />
      )}
    </>
  );
}

export default App;