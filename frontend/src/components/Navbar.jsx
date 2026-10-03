function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <a href="#" className="brand">
          <span className="brand-mark">AI</span>
          <span>Sentiment Analyzer</span>
        </a>

        <div className="nav-links">
          <a href="#analyzer">Analyzer</a>
          <a href="#model">Model</a>
          <a href="#pipeline">Pipeline</a>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;