import asyncio
from evaluation.gveval.scorer import Scorer

async def main():
    # Initialize the Scorer
    scorer = Scorer()

    # Sample input data
    pred = ["A man is playing a guitar."]
    ref = ["A person is playing a musical instrument.", "A man is strumming a guitar."]
    img_path = "example/video1.mp4"  # Replace with your actual image path

    # Perform the evaluation
    score = await scorer.gveval(pred, ref, img=img_path, visual='vid', setting='ref-free', accr=True, resolution='low')

    # Print the score
    print("Evaluation Score:", score['final_score'])
    print("Reason", score['reason'])

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
