import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

test.describe('FHE-GBDT Performance Case Study', () => {
    test('should display benchmark results and capture metrics', async ({ page }) => {
        // Navigate to the dashboard
        const dashboardPath = path.join(__dirname, '../dashboard.html');
        await page.goto(`file://${dashboardPath}`);

        // Wait for loading to complete
        await expect(page.locator('#loading')).toBeHidden({ timeout: 5000 });
        await expect(page.locator('#results-container')).toBeVisible();

        // Verify key metrics are displayed
        await expect(page.locator('#p50-latency')).toContainText('ms');
        await expect(page.locator('#p95-latency')).toContainText('ms');
        await expect(page.locator('#throughput')).toContainText('eps');

        // Verify table is populated
        const tableRows = page.locator('#results-table tr');
        await expect(tableRows).toHaveCount(4);

        // Verify stage timings are visible
        await expect(page.locator('text=t_step_bundle')).toBeVisible();
        await expect(page.locator('text=t_delta_linear')).toBeVisible();

        // Verify crypto stats
        await expect(page.locator('#rotations')).toContainText('12');
        await expect(page.locator('#switches')).toContainText('4');
        await expect(page.locator('#bootstraps')).toContainText('0');

        // Capture screenshot
        await page.screenshot({
            path: 'bench/reports/performance_case_study.png',
            fullPage: true
        });

        // Extract metrics for report
        const p50 = await page.locator('#p50-latency').textContent();
        const p95 = await page.locator('#p95-latency').textContent();
        const throughput = await page.locator('#throughput').textContent();

        const metricsReport = {
            timestamp: new Date().toISOString(),
            summary: {
                p50_latency: p50,
                p95_latency: p95,
                throughput: throughput,
            },
            test_status: 'PASSED',
            screenshot: 'bench/reports/performance_case_study.png'
        };

        fs.writeFileSync(
            'bench/reports/e2e_metrics.json',
            JSON.stringify(metricsReport, null, 2)
        );

        console.log('Performance Case Study Results:');
        console.log(`  P50 Latency: ${p50}`);
        console.log(`  P95 Latency: ${p95}`);
        console.log(`  Throughput:  ${throughput}`);
    });

    test('should verify SLO compliance', async ({ page }) => {
        const dashboardPath = path.join(__dirname, '../dashboard.html');
        await page.goto(`file://${dashboardPath}`);
        await expect(page.locator('#results-container')).toBeVisible({ timeout: 5000 });

        // Check SLO badge
        await expect(page.locator('.status-badge')).toContainText('All Tests Passing');

        // Verify error rate is within SLO
        await expect(page.locator('text=0.00%')).toBeVisible();
        await expect(page.locator('text=Within SLO')).toBeVisible();
    });
});
