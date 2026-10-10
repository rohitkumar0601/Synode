"""initial: dataset, run, job

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("target", sa.String(), nullable=False),
        sa.Column("n_rows", sa.Integer(), nullable=True),
        sa.Column("metadata_json", sa.String(), nullable=False),
        sa.Column("fingerprint", sa.String(), nullable=True),
        sa.Column("registered", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "run",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("dataset_id", sa.String(), sa.ForeignKey("dataset.id"), nullable=False),
        sa.Column("generators_json", sa.String(), nullable=False),
        sa.Column("seeds_json", sa.String(), nullable=False),
        sa.Column("preset", sa.String(), nullable=False),
        sa.Column("weights_json", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_run_dataset_id", "run", ["dataset_id"])
    op.create_table(
        "job",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("run.id"), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("generator", sa.String(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_job_run_id", "job", ["run_id"])


def downgrade() -> None:
    op.drop_table("job")
    op.drop_table("run")
    op.drop_table("dataset")
