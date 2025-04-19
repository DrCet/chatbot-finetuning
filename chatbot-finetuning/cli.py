import click

@click.group()
def cli():
    """Chatbot Finetuning: A pipeline for easy fine-tuning of chat models."""
    pass

@cli.command()
def init():
    """Initialize the pipeline environment."""
    click.echo("Initializing chatbot-finetuning pipeline...")

@cli.command()
@click.option('--model', default='meta-llama/Llama-3-8b', help='Hugging Face model name')
@click.option('--dataset', help='Path to dataset or Hugging Face dataset name')
def select(model, dataset):
    """Select model and dataset for fine-tuning."""
    click.echo(f"Selected model: {model}")
    click.echo(f"Selected dataset: {dataset}")

@cli.command()
@click.option('--output-dir', default='./finetuned_model', help='Output directory for fine-tuned model')
def train(output_dir):
    """Fine-tune the selected model."""
    click.echo(f"Training model and saving to {output_dir}...")

if __name__ == '__main__':
    cli()