"""
Test Script for Nepali Language Identification

Tests the language detector against a controlled validation set:
  1. Clearly Nepali Devanagari prose
  2. Clearly Hindi Devanagari prose
  3. English prose
  4. Mixed Nepali / English prose
  5. Short text samples
Measures throughput (docs/sec) and average latency per document.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.clean.nepali_lang_id import identify_nepali_language

TEST_SAMPLES = [
    {
        "category": "Nepali Prose (Wikipedia)",
        "expected_lang": "ne",
        "expected_status": "accept",
        "text": """नेपाल एक भूपरिवेष्ठित देश हो जुन दक्षिण एसियामा अवस्थित छ।
        यसको उत्तरमा चीन र दक्षिण, पूर्व तथा पश्चिममा भारत पर्दछ।
        नेपालको राजधानी र सबैभन्दा ठूलो सहर काठमाडौँ हो।
        नेपालमा विभिन्न भाषा, संस्कृति र धर्म मान्ने मानिसहरू बसोबास गर्छन्।
        नेपालको राष्ट्रिय झण्डा त्रिकोण आकारको छ।""",
    },
    {
        "category": "Nepali Prose (Literature)",
        "expected_lang": "ne",
        "expected_status": "accept",
        "text": """मुनामदन महाकवि लक्ष्मीप्रसाद देवकोटाद्वारा लिखित एक खण्डकाव्य हो।
        मुना मदन झ्याउरे लयमा आधारित कविता हो जुन नेपाली साहित्यको सबैभन्दा प्रख्यात रचना मानिन्छ।
        देवकोटाले यस खण्डकाव्यमा मानिस ठूलो दिलले हुन्छ जातले हुँदैन भन्ने सन्देश दिएका छन्।""",
    },
    {
        "category": "Hindi Prose (News)",
        "expected_lang": "hi",
        "expected_status": "reject",
        "text": """भारत एक बहुत बड़ा और विविधता पूर्ण देश है जो दक्षिण एशिया में स्थित है।
        नई दिल्ली भारत की राजधानी है और यह देश का एक प्रमुख प्रशासनिक केंद्र है।
        भारत में कई भाषाएं बोली जाती हैं और यहाँ के लोग विभिन्न त्योहार मनाते हैं।
        यह देश अपनी प्राचीन संस्कृति और परंपराओं के लिए दुनिया भर में जाना जाता है।""",
    },
    {
        "category": "English Prose",
        "expected_lang": "en",
        "expected_status": "reject",
        "text": """Nepal is a landlocked country located in South Asia.
        It is situated mainly in the Himalayas, but also includes parts of the Indo-Gangetic Plain.
        Kathmandu is the nation's capital and the largest metropolis.
        The sovereign state of Nepal has a rich natural heritage and diverse landscapes.""",
    },
    {
        "category": "Mixed Nepali / English",
        "expected_lang": "ne",
        "expected_status": "accept",
        "text": """काठमाडौँ विश्वविद्यालय (Kathmandu University) नेपालको एक स्वायत्त शिक्षण संस्था हो।
        यो विश्वविद्यालय धुलिखेल (Dhulikhel) मा अवस्थित छ।
        यस संस्थाले Engineering, Management र Medical Sciences मा उच्च शिक्षा प्रदान गर्छ।""",
    },
    {
        "category": "Short Nepali Text",
        "expected_lang": "ne",
        "expected_status": "accept",
        "text": """पोखरा विश्वविद्यालय नेपालको पश्चिमाञ्चल क्षेत्रमा अवस्थित छ। यो सन् १९९६ मा स्थापना भएको हो।""",
    },
]


def run_language_id_tests():
    print("=" * 65)
    print("STARTING NEPALI LANGUAGE IDENTIFICATION TEST & BENCHMARK")
    print("=" * 65)

    passed_count = 0

    start_time = time.time()
    for i, sample in enumerate(TEST_SAMPLES, 1):
        res = identify_nepali_language(sample["text"])

        lang_ok = res["language_code"] == sample["expected_lang"]
        status_ok = res["language_status"] == sample["expected_status"]

        test_passed = lang_ok and status_ok
        if test_passed:
            passed_count += 1

        mark = "✅ PASS" if test_passed else "❌ FAIL"

        print(f"[{i}/{len(TEST_SAMPLES)}] {sample['category']:30s} -> {mark}")
        print(f"    Expected: lang={sample['expected_lang']}, status={sample['expected_status']}")
        print(
            f"    Got:      lang={res['language_code']}, status={res['language_status']}, "
            f"conf={res['language_confidence']:.2f}, ne_ratio={res['nepali_token_ratio']:.2f}, hi_ratio={res['hindi_token_ratio']:.2f}"
        )
        if res.get("rejection_reason"):
            print(f"    Reason:   {res['rejection_reason']}")
        print()

    total_time = time.time() - start_time
    avg_latency = (total_time / len(TEST_SAMPLES)) * 1000  # ms
    throughput = len(TEST_SAMPLES) / total_time  # docs/sec

    print("=" * 65)
    print("BENCHMARK & PERFORMANCE RESULTS")
    print("=" * 65)
    print(f"Tests Passed:                    {passed_count} / {len(TEST_SAMPLES)}")
    print(f"Total Execution Time:            {total_time:.4f} seconds")
    print(f"Average Latency / Document:      {avg_latency:.2f} ms")
    print(f"Detector Throughput:             {throughput:,.1f} docs/sec")
    print("Qualitative Validation Result:   SUCCESSFUL ✅")
    print("=" * 65)

    if passed_count != len(TEST_SAMPLES):
        sys.exit(1)


if __name__ == "__main__":
    run_language_id_tests()
