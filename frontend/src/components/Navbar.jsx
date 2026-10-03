function Navbar() {
  return (
    <nav
      className="navbar"
      aria-label="Main navigation"
    >
      <div className="navbar-inner">
        <a
          href="#"
          className="brand"
          aria-label="Sentiment Analyzer home"
        >
          <span
            className="brand-mark"
            aria-hidden="true"
          >
            AI
          </span>

          <span>Sentiment Analyzer</span>
        </a>

        <div className="nav-links">
          <a href="#analyzer">
            Analyzer
          </a>

          <a href="#model">
            Model
          </a>

          <a href="#pipeline">
            Pipeline
          </a>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;