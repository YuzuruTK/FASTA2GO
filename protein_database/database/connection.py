"""Database connection management for the protein feature database."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from protein_database.database.models import Base


def create_database_engine(db_path: str | Path) -> Engine:
    """Create a SQLAlchemy engine for the SQLite database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        SQLAlchemy Engine instance.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    return engine


def init_database(engine: Engine) -> None:
    """Create all tables in the database if they don't exist.

    Args:
        engine: SQLAlchemy Engine instance.
    """
    Base.metadata.create_all(engine)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to the given engine.

    Args:
        engine: SQLAlchemy Engine instance.

    Returns:
        A sessionmaker bound to the engine.
    """
    return sessionmaker(bind=engine, expire_on_commit=False)