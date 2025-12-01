// src/components/BackgroundPattern.tsx
export default function BackgroundPattern() {
  return (
    <>
      {/* Dot Grid */}
      <div className="absolute inset-0 pointer-events-none opacity-30">
        <div 
          className="w-full h-full"
          style={{
            backgroundImage: 'radial-gradient(circle, #d1d5db 1px, transparent 1px)',
            backgroundSize: '30px 30px',
          }}
        />
      </div>

      {/* Geometric Shapes */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none opacity-20">
        {/* Circles - Green */}
        <div className="absolute top-24 right-32 w-2 h-2 bg-green-600 rounded-full" />
        <div className="absolute top-40 right-56 w-3 h-3 bg-green-500 rounded-full" />
        <div className="absolute top-32 right-72 w-2 h-2 bg-green-600 rounded-full" />
        
        {/* Circles - Gray */}
        <div className="absolute bottom-40 left-24 w-2 h-2 bg-gray-600 rounded-full" />
        <div className="absolute bottom-56 left-48 w-3 h-3 bg-gray-500 rounded-full" />
        
        {/* Triangles */}
        <div 
          className="absolute top-48 left-16 w-24 h-24 border-2 border-green-500 transform rotate-12"
          style={{ clipPath: 'polygon(50% 0%, 0% 100%, 100% 100%)' }}
        />
        <div 
          className="absolute bottom-32 right-24 w-32 h-32 border-2 border-gray-500 transform -rotate-45"
          style={{ clipPath: 'polygon(50% 0%, 0% 100%, 100% 100%)' }}
        />
        
        {/* Diagonal Lines */}
        <div className="absolute top-1/3 right-16 flex gap-1 transform rotate-45">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="w-0.5 h-16 bg-green-400" />
          ))}
        </div>
        
        <div className="absolute bottom-1/4 left-32 flex gap-1 transform -rotate-45">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="w-0.5 h-12 bg-gray-400" />
          ))}
        </div>
      </div>
    </>
  );
}