/* This is auto-generated code. Edit at your own peril. */
#include <stdio.h>
#include <stdlib.h>

#include "CuTest.h"


extern void TestCuStringNew(CuTest*);
extern void TestCuStringAppend(CuTest*);
extern void TestCuStringAppendNULL(CuTest*);
extern void TestCuStringAppendChar(CuTest*);
extern void TestCuStringInserts(CuTest*);
extern void TestCuStringResizes(CuTest*);
extern void TestPasses(CuTest*);
extern void TestCuTestNew(CuTest*);
extern void TestCuTestInit(CuTest*);
extern void TestCuAssert(CuTest*);
extern void TestCuAssertPtrEquals_Success(CuTest*);
extern void TestCuAssertPtrEquals_Failure(CuTest*);
extern void TestCuAssertPtrNotNull_Success(CuTest*);
extern void TestCuAssertPtrNotNull_Failure(CuTest*);
extern void TestCuTestRun(CuTest*);
extern void TestCuSuiteInit(CuTest*);
extern void TestCuSuiteNew(CuTest*);
extern void TestCuSuiteAddTest(CuTest*);
extern void TestCuSuiteAddSuite(CuTest*);
extern void TestCuSuiteRun(CuTest*);
extern void TestCuSuiteSummary(CuTest*);
extern void TestCuSuiteDetails_SingleFail(CuTest*);
extern void TestCuSuiteDetails_SinglePass(CuTest*);
extern void TestCuSuiteDetails_MultiplePasses(CuTest*);
extern void TestCuSuiteDetails_MultipleFails(CuTest*);
extern void TestCuStrCopy(CuTest*);
extern void TestCuStringAppendFormat(CuTest*);
extern void TestFail(CuTest*);
extern void TestAssertStrEquals(CuTest*);
extern void TestAssertStrEquals_NULL(CuTest*);
extern void TestAssertStrEquals_FailNULLStr(CuTest*);
extern void TestAssertStrEquals_FailStrNULL(CuTest*);
extern void TestAssertIntEquals(CuTest*);
extern void TestAssertDblEquals(CuTest*);
extern void Test_deviceInfo_gather(CuTest*);


void RunAllTests(void) 
{
    CuString *output = CuStringNew();
    CuSuite* suite = CuSuiteNew();


    SUITE_ADD_TEST(suite, TestCuStringNew);
    SUITE_ADD_TEST(suite, TestCuStringAppend);
    SUITE_ADD_TEST(suite, TestCuStringAppendNULL);
    SUITE_ADD_TEST(suite, TestCuStringAppendChar);
    SUITE_ADD_TEST(suite, TestCuStringInserts);
    SUITE_ADD_TEST(suite, TestCuStringResizes);
    SUITE_ADD_TEST(suite, TestPasses);
    SUITE_ADD_TEST(suite, TestCuTestNew);
    SUITE_ADD_TEST(suite, TestCuTestInit);
    SUITE_ADD_TEST(suite, TestCuAssert);
    SUITE_ADD_TEST(suite, TestCuAssertPtrEquals_Success);
    SUITE_ADD_TEST(suite, TestCuAssertPtrEquals_Failure);
    SUITE_ADD_TEST(suite, TestCuAssertPtrNotNull_Success);
    SUITE_ADD_TEST(suite, TestCuAssertPtrNotNull_Failure);
    SUITE_ADD_TEST(suite, TestCuTestRun);
    SUITE_ADD_TEST(suite, TestCuSuiteInit);
    SUITE_ADD_TEST(suite, TestCuSuiteNew);
    SUITE_ADD_TEST(suite, TestCuSuiteAddTest);
    SUITE_ADD_TEST(suite, TestCuSuiteAddSuite);
    SUITE_ADD_TEST(suite, TestCuSuiteRun);
    SUITE_ADD_TEST(suite, TestCuSuiteSummary);
    SUITE_ADD_TEST(suite, TestCuSuiteDetails_SingleFail);
    SUITE_ADD_TEST(suite, TestCuSuiteDetails_SinglePass);
    SUITE_ADD_TEST(suite, TestCuSuiteDetails_MultiplePasses);
    SUITE_ADD_TEST(suite, TestCuSuiteDetails_MultipleFails);
    SUITE_ADD_TEST(suite, TestCuStrCopy);
    SUITE_ADD_TEST(suite, TestCuStringAppendFormat);
    SUITE_ADD_TEST(suite, TestFail);
    SUITE_ADD_TEST(suite, TestAssertStrEquals);
    SUITE_ADD_TEST(suite, TestAssertStrEquals_NULL);
    SUITE_ADD_TEST(suite, TestAssertStrEquals_FailNULLStr);
    SUITE_ADD_TEST(suite, TestAssertStrEquals_FailStrNULL);
    SUITE_ADD_TEST(suite, TestAssertIntEquals);
    SUITE_ADD_TEST(suite, TestAssertDblEquals);
    SUITE_ADD_TEST(suite, Test_deviceInfo_gather);

    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s\n", output->buffer);
    CuStringDelete(output);
    CuSuiteDelete(suite);
}

int main(void)
{
    RunAllTests();
}

